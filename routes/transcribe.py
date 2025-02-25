from models import TranscriptionRequest, TranscriptionResponse
import tempfile
import random
import time
import os
import math
import subprocess
import asyncio
import uuid
from typing import Optional

from fastapi import File, UploadFile, BackgroundTasks, HTTPException, APIRouter

router = APIRouter()


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: Optional[TranscriptionRequest] = None,
):
    """
    Transcribe an audio file using OpenAI's Whisper API in a streaming buffer approach.

    The audio will be processed in chunks as soon as they're available, without waiting
    for the entire file to be split.
    """
    start_time = time.time()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded")

    # Create temp file for the uploaded audio
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename or "")[1]
    )
    temp_file_path = temp_file.name
    temp_file.close()

    # Save uploaded file to temp file
    logger.info(f"Saving uploaded file: {file.filename}")
    async with aiofiles.open(temp_file_path, "wb") as out_file:
        # Process in chunks to handle large files more efficiently
        chunk_size = 1024 * 1024  # 1MB chunks for file reading
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            await out_file.write(chunk)

    logger.info(f"File saved to {temp_file_path}, beginning processing")

    # Create a session directory for chunks
    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Set up queues for producer/consumer pattern
    # This allows us to start transcribing chunks as soon as they are created
    chunk_queue = asyncio.Queue()
    result_queue = asyncio.Queue()

    # Create asyncio Event to signal when splitting is complete
    splitting_complete = asyncio.Event()

    # Track the order of chunks
    chunk_order = {}

    async def split_audio_in_background():
        """Split audio file into chunks and enqueue them as they are created"""
        try:
            # Get audio information
            duration_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                temp_file_path,
            ]

            logger.info("Getting audio duration")
            await result_queue.put(
                {"status": "update", "message": "Analyzing audio file..."}
            )

            duration_process = subprocess.run(
                duration_cmd, capture_output=True, text=True
            )
            if duration_process.returncode != 0:
                logger.error("Failed to get audio duration")
                raise HTTPException(
                    status_code=500, detail="Error getting audio duration"
                )

            total_duration = float(duration_process.stdout.strip())
            logger.info(f"Audio duration: {total_duration:.2f} seconds")
            await result_queue.put(
                {
                    "status": "update",
                    "message": f"Audio duration: {total_duration:.2f} seconds",
                }
            )

            # Calculate optimal chunk duration (around 30 seconds to 1 minute per chunk)
            # Smaller chunks start processing faster
            chunk_duration = min(
                60, max(30, total_duration / 20)
            )  # At least 20 chunks or 30-60s each

            chunk_count = math.ceil(total_duration / chunk_duration)
            logger.info(
                f"Splitting into approximately {chunk_count} chunks of {chunk_duration:.2f}s each"
            )
            await result_queue.put(
                {
                    "status": "update",
                    "message": f"Splitting into {chunk_count} chunks of {chunk_duration:.2f}s each",
                    "total_chunks": chunk_count,
                }
            )

            # Push event to result queue instead of yielding directly
            splitting_event = {
                "status": "splitting",
                "total_chunks": chunk_count,
                "duration": total_duration,
            }
            logger.info(f"Queuing splitting event: {splitting_event}")
            await result_queue.put(splitting_event)

            # Create chunks
            for i in range(0, math.ceil(total_duration / chunk_duration)):
                start_time_sec = i * chunk_duration
                duration_sec = min(chunk_duration, total_duration - start_time_sec)

                if duration_sec <= 0:
                    break

                # Send update that we're starting to create this chunk
                creating_event = {
                    "status": "update",
                    "message": f"Creating chunk {i + 1}/{chunk_count}",
                    "stage": "creating_chunk",
                    "current_chunk": i + 1,
                    "total_chunks": chunk_count,
                }
                logger.info(f"Queuing chunk creation event: {creating_event}")
                await result_queue.put(creating_event)

                output_path = str(session_dir / f"chunk_{i:03d}.mp3")

                # Use ffmpeg to split
                cmd = [
                    "ffmpeg",
                    "-i",
                    temp_file_path,
                    "-ss",
                    str(start_time_sec),
                    "-t",
                    str(duration_sec),
                    "-c:a",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    "-af",
                    "apad=pad_dur=0.5",
                    output_path,
                ]

                logger.info(f"Creating chunk {i + 1}/{chunk_count}")
                process = subprocess.run(cmd, capture_output=True)

                if process.returncode != 0:
                    error_msg = (
                        process.stderr.decode("utf-8", errors="replace")
                        if process.stderr
                        else "Unknown error"
                    )
                    logger.error(f"Error creating chunk {i}: {error_msg}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error splitting audio: {error_msg}",
                    )

                # Track chunk order
                chunk_order[output_path] = i

                # Put chunk in the queue for processing
                await chunk_queue.put(output_path)

                # Report progress by pushing to result queue
                chunk_created_event = {
                    "status": "chunk_created",
                    "chunk": i + 1,
                    "total_chunks": chunk_count,
                }
                logger.info(f"Queuing chunk created event: {chunk_created_event}")
                await result_queue.put(chunk_created_event)

            logger.info("All chunks created and enqueued")
            await result_queue.put(
                {
                    "status": "update",
                    "message": "All chunks created, completing transcription...",
                }
            )

        except Exception as e:
            logger.error(f"Error in split_audio_in_background: {str(e)}")
            # Re-raise to be caught by the outer try-except
            raise
        finally:
            # Signal that splitting is complete
            splitting_complete.set()

    async def process_chunks():
        """Process chunks from the queue as they become available"""
        active_tasks = set()
        processed_chunks = set()

        try:
            while True:
                # If queue is empty and splitting is complete and no active tasks, we're done
                if (
                    chunk_queue.empty()
                    and splitting_complete.is_set()
                    and not active_tasks
                ):
                    logger.info(
                        f"All chunks processed. Total processed: {len(processed_chunks)}"
                    )
                    break

                # Try to get a chunk from the queue, with a short timeout
                try:
                    chunk_path = await asyncio.wait_for(chunk_queue.get(), timeout=0.1)
                    chunk_idx = chunk_order[chunk_path]

                    # Start an independent task to process this chunk
                    async def process_chunk(path, idx):
                        try:
                            # Update that we're starting to process this chunk
                            await result_queue.put(
                                {
                                    "status": "processing",
                                    "message": f"Transcribing chunk {idx + 1}",
                                    "chunk": idx + 1,
                                }
                            )

                            logger.info(f"Starting transcription of chunk {idx + 1}")
                            start_time = time.time()

                            # Process the chunk with Whisper API
                            chunk_text = await process_audio_chunk(
                                path,
                                language=request.language if request else None,
                                prompt=request.prompt if request else None,
                            )

                            # Calculate processing time
                            processing_time = time.time() - start_time

                            logger.info(
                                f"✓ Completed chunk {idx + 1} in {processing_time:.2f}s"
                            )
                            processed_chunks.add(idx)

                            # Send the result back
                            await result_queue.put(
                                {
                                    "status": "chunk_result",
                                    "chunk_index": idx,
                                    "total_chunks": len(chunk_order)
                                    if splitting_complete.is_set()
                                    else "unknown",
                                    "text": chunk_text.strip(),
                                    "processing_time": processing_time,
                                    "is_final": len(processed_chunks)
                                    == len(chunk_order)
                                    and splitting_complete.is_set(),
                                    "chunks_processed": len(processed_chunks),
                                    "chunks_total": len(chunk_order)
                                    if splitting_complete.is_set()
                                    else "in progress",
                                }
                            )

                            # Store result
                            await result_queue.put((idx, chunk_text.strip()))
                        except Exception as e:
                            logger.error(f"Error processing chunk {idx + 1}: {str(e)}")
                            # Put a None result to indicate error
                            await result_queue.put(
                                {
                                    "status": "chunk_error",
                                    "chunk_index": idx,
                                    "message": str(e),
                                }
                            )

                    # Create an independent task for this chunk
                    task = asyncio.create_task(process_chunk(chunk_path, chunk_idx))

                    # Add to active tasks set
                    active_tasks.add(task)

                    # Set up callback to remove from active tasks when done
                    def task_done(t):
                        active_tasks.discard(t)

                    task.add_done_callback(task_done)

                except asyncio.TimeoutError:
                    # No chunk available yet, check if we should continue waiting
                    if not splitting_complete.is_set() or active_tasks:
                        # Send a status update every 10 iterations
                        if random.random() < 0.1:  # ~10% chance to send update
                            stats = {
                                "active_tasks": len(active_tasks),
                                "processed": len(processed_chunks),
                                "total_known": len(chunk_order),
                            }
                            await result_queue.put(
                                {
                                    "status": "update",
                                    "message": f"Processing {len(active_tasks)} chunks in parallel. Completed {len(processed_chunks)}/{len(chunk_order) if chunk_order else '?'} chunks.",
                                    "stats": stats,
                                }
                            )
                        continue
                    else:
                        # Nothing left to process
                        break

            # Send a final update with stats
            await result_queue.put(
                {
                    "status": "update",
                    "message": f"Completed transcription of all {len(processed_chunks)} chunks",
                    "stats": {
                        "processed": len(processed_chunks),
                        "total": len(chunk_order),
                    },
                }
            )

            # Final transcription progress update
            if chunk_order:
                final_progress = {
                    "status": "progress_update",
                    "message": "Transcription complete",
                    "chunking_progress": {
                        "created": len(chunk_order),
                        "total": len(chunk_order),
                    },
                    "transcription_progress": {
                        "processed": len(processed_chunks),
                        "total": len(chunk_order),
                    },
                }
                await result_queue.put(final_progress)

            # Signal that processor task is complete (this was missing)
            await result_queue.put({"status": "task_complete", "task": "processor"})

            # Sort results by chunk index
            logger.info(f"All chunks processed: {len(processed_chunks)} chunks")
            results = sorted(await result_queue.get())
            transcription_text = "\n".join([text for _, text in results])

            # Return the results
            return transcription_text, results

        except Exception as e:
            logger.error(f"Error in process_chunks: {str(e)}")
            await result_queue.put(
                {"status": "error", "message": f"Error in chunk processing: {str(e)}"}
            )

            # Signal that processor task is complete even on error
            await result_queue.put({"status": "task_complete", "task": "processor"})

            # Return empty results on error
            return "", []
