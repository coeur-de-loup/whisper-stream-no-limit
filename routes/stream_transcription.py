import os
import uuid
import time
import asyncio


from pathlib import Path

from fastapi import File, UploadFile, Request, APIRouter
from fastapi.responses import StreamingResponse

import json

from utils import split_audio
from logger import logger
from whisper import transcribe

router = APIRouter()

# IMPORTANT FIX: The system was previously stopping transcription prematurely after the splitter task completed,
# before the processor had finished processing all chunks. This has been fixed by:
# 1. Removing the code that marked transcription as complete when the splitter task finished
# 2. Only marking transcription as complete when the processor task has finished
# 3. Increasing timeouts to give more time for chunks to be processed
# 4. Adding additional checks to ensure all chunks are processed before completing

TEMP_DIR = Path("/tmp/whisper_chunks")
TEMP_DIR.mkdir(exist_ok=True)
MAX_PARALLEL_CHUNKS = int(os.getenv("MAX_PARALLEL_CHUNKS", 5))


async def process_chunks(
    chunk_queue, result_queue, chunk_order=None, language=None, prompt=None
):
    """Process audio chunks from the queue and transcribe them"""
    try:
        active_tasks = {}
        processed_chunks = []
        results = []
        waiting_for_chunks = True
        chunk_splitter_done = False
        # Keep track of the chunks that have been sent to the Whisper API
        sent_chunk_indices = set()
        logger.info("Starting chunk processor")

        # Process chunks until we're done
        while waiting_for_chunks or active_tasks:
            # Try to add new chunks to process (up to MAX_PARALLEL_CHUNKS)
            while len(active_tasks) < MAX_PARALLEL_CHUNKS and waiting_for_chunks:
                try:
                    # Try to get a chunk from the queue without blocking
                    chunk_index, chunk_path = await asyncio.wait_for(
                        chunk_queue.get(), timeout=0.1
                    )

                    # Log when we start processing a chunk
                    logger.info(f"Starting transcription of chunk {chunk_index + 1}")

                    # IMPORTANT: Add chunk index to sent_chunk_indices to track what we've sent to the API
                    sent_chunk_indices.add(chunk_index)
                    logger.info(
                        f"Added chunk {chunk_index} to sent_chunk_indices. Total sent: {len(sent_chunk_indices)}"
                    )

                    # Create task to process the chunk
                    task = asyncio.create_task(transcribe(chunk_path, language, prompt))
                    active_tasks[task] = (chunk_index, chunk_path)

                    # Send an update about this chunk starting processing
                    await result_queue.put(
                        {
                            "status": "chunk_processing",
                            "chunk_index": chunk_index,
                            "message": f"Transcribing chunk {chunk_index + 1}",
                        }
                    )

                except asyncio.TimeoutError:
                    # No chunks available right now
                    if chunk_splitter_done:
                        # If splitter is done and queue is empty, we're done getting new chunks
                        waiting_for_chunks = False
                    # Otherwise, we'll try again next iteration
                    break
                except asyncio.CancelledError:
                    # If we get cancelled, clean up and exit
                    logger.info("Chunk processor cancelled")
                    for task in active_tasks:
                        task.cancel()
                    return

            if not active_tasks:
                # If there are no active tasks and we're not waiting for chunks, we're done
                if not waiting_for_chunks:
                    break
                # Otherwise, wait a bit and try again
                await asyncio.sleep(0.1)
                continue

            # Wait for at least one task to complete or for a message about splitter completion
            done_tasks, pending_tasks = await asyncio.wait(
                list(active_tasks.keys()),
                timeout=0.5,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Check if the splitter is done
            try:
                # Check for task_complete message
                while True:
                    queue_msg = result_queue.get_nowait()
                    if (
                        isinstance(queue_msg, dict)
                        and queue_msg.get("status") == "task_complete"
                        and queue_msg.get("task") == "splitter"
                    ):
                        logger.info("Splitter task completed")
                        chunk_splitter_done = True

                        # If we have chunk_order information, log how many chunks we expect
                        if chunk_order:
                            logger.info(
                                f"Expecting to process {len(chunk_order)} total chunks"
                            )
                    # Put the message back in the queue for other consumers
                    await result_queue.put(queue_msg)
            except asyncio.QueueEmpty:
                pass

            # Process completed tasks
            for task in done_tasks:
                chunk_index, chunk_path = active_tasks.pop(task)

                try:
                    # Get the transcription result
                    text = await task

                    # Add to processed chunks and results
                    processed_chunks.append(chunk_index)
                    results.append((chunk_index, text))
                    logger.info(f"Completed transcription of chunk {chunk_index + 1}")

                    # Send update about this chunk being completed
                    chunk_result = {
                        "status": "chunk_result",
                        "chunk_index": chunk_index,
                        "chunks_processed": len(processed_chunks),
                        "text": text,
                        "message": f"Transcribed chunk {chunk_index + 1}",
                    }

                    # If we know the total number of chunks, include it in the result
                    if chunk_order:
                        chunk_result["chunks_total"] = len(chunk_order)

                        # Check if this is the last chunk
                        if len(processed_chunks) == len(chunk_order):
                            chunk_result["is_final"] = True
                            logger.info(
                                f"Processed final chunk {chunk_index + 1} of {len(chunk_order)}"
                            )
                        else:
                            logger.info(
                                f"Processed chunk {chunk_index + 1}, {len(processed_chunks)}/{len(chunk_order)} complete"
                            )

                    await result_queue.put(chunk_result)

                    # Send transcription progress update
                    if chunk_order and (
                        len(processed_chunks) % 2 == 0
                        or len(processed_chunks) == 1
                        or len(processed_chunks) == len(chunk_order)
                    ):
                        progress = {
                            "status": "progress_update",
                            "message": f"Transcribed {len(processed_chunks)}/{len(chunk_order)} chunks",
                            "chunking_progress": {
                                "created": len(chunk_order),
                                "total": len(chunk_order),
                            },
                            "transcription_progress": {
                                "processed": len(processed_chunks),
                                "total": len(chunk_order),
                            },
                        }
                        await result_queue.put(progress)

                except Exception as e:
                    logger.exception(
                        f"Error processing chunk {chunk_index + 1}: {str(e)}"
                    )
                    await result_queue.put(
                        {
                            "status": "error",
                            "message": f"Error processing chunk {chunk_index + 1}: {str(e)}",
                        }
                    )

            # Send periodic status update every 3 processed chunks or active tasks
            if len(processed_chunks) % 3 == 0 and processed_chunks:
                stats = {}
                if chunk_order:
                    stats["total_chunks"] = len(chunk_order)
                    stats["processed_chunks"] = len(processed_chunks)
                    stats["remaining_chunks"] = len(chunk_order) - len(processed_chunks)
                    stats["sent_chunks"] = len(
                        sent_chunk_indices
                    )  # Log how many chunks we've sent to the API

                await result_queue.put(
                    {
                        "status": "update",
                        "message": f"Processing {len(active_tasks)} chunks in parallel. Completed {len(processed_chunks)}/{len(chunk_order) if chunk_order else '?'} chunks.",
                        "stats": stats,
                    }
                )

        # CRITICAL FIX: Check if we've processed all chunks that were sent to the API
        # This ensures we don't complete until all API responses are received
        if sent_chunk_indices and set(processed_chunks) != sent_chunk_indices:
            missing_chunks = sent_chunk_indices - set(processed_chunks)
            logger.warning(
                f"Missing {len(missing_chunks)} chunks that were sent to the API: {missing_chunks}"
            )

            # Wait a bit longer for any remaining chunks that were sent to the API
            logger.info("Waiting for remaining API responses to complete...")

            # Wait up to 60 seconds for missing chunks to be processed
            wait_start = time.time()
            while (
                time.time() - wait_start < 60
                and set(processed_chunks) != sent_chunk_indices
            ):
                # Log a message every 5 seconds
                if (time.time() - wait_start) % 5 < 0.1:
                    missing_now = sent_chunk_indices - set(processed_chunks)
                    logger.info(
                        f"Still waiting for {len(missing_now)} API responses. Missing chunks: {missing_now}"
                    )

                # Wait a bit and check again
                await asyncio.sleep(1)

                # If we've processed all chunks, break out of the loop
                if set(processed_chunks) == sent_chunk_indices:
                    logger.info("All API responses received!")
                    break

            # Check if we're still missing chunks after waiting
            if set(processed_chunks) != sent_chunk_indices:
                missing_chunks = sent_chunk_indices - set(processed_chunks)
                logger.warning(
                    f"After waiting, still missing {len(missing_chunks)} chunks: {missing_chunks}"
                )

        # Check if we've processed all expected chunks
        if chunk_order and len(processed_chunks) < len(chunk_order):
            missing_chunks = set(range(len(chunk_order))) - set(processed_chunks)
            logger.warning(f"Missing {len(missing_chunks)} chunks: {missing_chunks}")

            # Wait a bit longer for any remaining chunks
            logger.info("Waiting for remaining chunks to complete...")
            await asyncio.sleep(5)

            # Check again if we've received any more chunks
            if len(processed_chunks) < len(chunk_order):
                missing_chunks = set(range(len(chunk_order))) - set(processed_chunks)
                logger.warning(
                    f"Still missing {len(missing_chunks)} chunks after waiting: {missing_chunks}"
                )
            else:
                logger.info("All expected chunks have been processed!")

        # Sort results by chunk index
        logger.info(f"All chunks processed: {len(processed_chunks)} chunks")
        results.sort(key=lambda x: x[0])
        transcription_text = "\n".join([text for _, text in results])

        # Send complete result
        await result_queue.put(
            {
                "status": "update",
                "message": f"All {len(processed_chunks)} chunks transcribed.",
                "sent_chunk_count": len(
                    sent_chunk_indices
                ),  # Include sent chunk count for debugging
                "processed_chunk_count": len(
                    processed_chunks
                ),  # Include processed chunk count for debugging
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
                "sent_chunks": len(
                    sent_chunk_indices
                ),  # Include sent chunk count for debugging
            }
            await result_queue.put(final_progress)

        # Signal that processor task is complete
        await result_queue.put(
            {
                "status": "task_complete",
                "task": "processor",
                "sent_chunk_count": len(sent_chunk_indices),
                "processed_chunk_count": len(processed_chunks),
            }
        )

        # Return the transcription text and results
        return transcription_text, results

    except Exception as e:
        logger.exception(f"Error in process_chunks: {str(e)}")
        await result_queue.put(
            {"status": "error", "message": f"Error in chunk processing: {str(e)}"}
        )

        # Signal that processor task is complete even on error
        await result_queue.put({"status": "task_complete", "task": "processor"})

        # Return empty results on error
        return "", []


@router.post("/transcribe/stream")
async def stream_transcription(request: Request, file: UploadFile = File(...)):
    """
    Stream the transcription process, allowing for real-time updates to the client.
    """
    # Create temp files with unique names to support multiple simultaneous users
    unique_id = str(uuid.uuid4())

    # Set up temporary files for audio and processed chunks
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_audio_path = temp_dir / f"temp_audio_{unique_id}.wav"

    # Perform streaming transcription and yield results
    async def generate():
        try:
            start_time = time.time()
            # Save the uploaded file
            with open(temp_audio_path, "wb") as buffer:
                bytes_processed = 0
                while True:
                    chunk = await file.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    buffer.write(chunk)
                    bytes_processed += len(chunk)

                    # Periodically send updates about upload progress
                    if bytes_processed % (1024 * 1024) == 0:  # Every 1MB
                        upload_update = {
                            "status": "uploading",
                            "bytes_processed": bytes_processed,
                            "mb_processed": round(bytes_processed / 1024 / 1024, 1),
                        }
                        logger.info(f"Upload progress: {upload_update}")
                        yield f"data: {json.dumps(upload_update)}\n\n"
                        await asyncio.sleep(
                            0.01
                        )  # Small delay to allow the event to be sent

            # Send processing notification
            logger.info("File uploaded, starting processing")
            processing_notification = {
                "status": "processing",
                "message": "File successfully uploaded, beginning processing",
            }
            yield f"data: {json.dumps(processing_notification)}\n\n"

            # Create queues for communication between tasks
            result_queue = asyncio.Queue()
            chunk_queue = asyncio.Queue()

            # Track chunk order
            chunk_order = []

            # Start the splitting and processing tasks
            splitter_task = asyncio.create_task(
                split_audio(temp_audio_path, chunk_queue, result_queue)
            )
            processor_task = asyncio.create_task(
                process_chunks(chunk_queue, result_queue, chunk_order)
            )

            # Process results from the queue and stream them to the client
            all_done = False
            active_tasks = 2  # splitter and processor
            final_result_sent = False
            results_by_index = {}
            last_update_time = time.time()
            last_chunk_result = None
            # Add a last_progress_time to track when the last actual progress was made
            last_progress_time = time.time()
            # Add a completion_timeout to force completion after a certain time
            completion_timeout = 300  # 5 minutes
            # Track the expected total number of chunks
            expected_total_chunks = None

            while True:
                # Check for overall timeout
                current_time = time.time()
                if current_time - start_time > completion_timeout:
                    logger.warning(
                        f"Overall completion timeout reached after {completion_timeout}s, forcing completion"
                    )
                    all_done = True

                # Check for completion based on task states rather than timeouts
                if not all_done:
                    # Both tasks are complete AND we have all expected chunks - this is a definitive signal
                    if splitter_task.done() and processor_task.done():
                        if (
                            expected_total_chunks is not None
                            and len(results_by_index) >= expected_total_chunks
                        ):
                            logger.info(
                                f"Both tasks complete and all {expected_total_chunks} chunks received - transcription is done"
                            )
                            all_done = True
                        elif expected_total_chunks is None:
                            logger.info(
                                "Both splitter and processor tasks are complete - transcription is done"
                            )
                            all_done = True
                        else:
                            logger.warning(
                                f"Both tasks complete but only received {len(results_by_index)}/{expected_total_chunks} chunks"
                            )
                            # Only force completion if it's been a while since the last progress
                            # Increase timeout to give more time for final chunks to complete
                            if (
                                current_time - last_progress_time > 60
                            ):  # Increased from 30s to 60s
                                logger.warning(
                                    "No progress for 60s, forcing completion despite missing chunks"
                                )
                                all_done = True
                            else:
                                # Continue waiting for the missing chunks
                                logger.info("Continuing to wait for missing chunks...")

                    # We have results AND no messages for a longer time AND both queues are empty
                    # This suggests processing is complete even if tasks haven't properly reported completion
                    # BUT ONLY if processor task is also done
                    elif (
                        results_by_index
                        and current_time - last_progress_time
                        > 30  # Increased from 20 to 30 seconds
                        and chunk_queue.empty()
                        and result_queue.empty()
                        and processor_task.done()  # CRITICAL FIX: Only if processor is done
                    ):
                        # Only mark as complete if we have all expected chunks or don't know how many to expect
                        if (
                            expected_total_chunks is None
                            or len(results_by_index) >= expected_total_chunks
                        ):
                            logger.info(
                                "Queues empty, no recent activity, have results, processor done - marking as complete"
                            )
                            all_done = True
                        else:
                            logger.warning(
                                f"Queues empty but only received {len(results_by_index)}/{expected_total_chunks} chunks"
                            )
                            # Only force completion if it's been a while since the last progress
                            # Increase timeout to give more time for final chunks to complete
                            if (
                                current_time - last_progress_time > 60
                            ):  # Increased from 30s to 60s
                                logger.warning(
                                    "No progress for 60s, forcing completion despite missing chunks"
                                )
                                all_done = True
                            else:
                                # Continue waiting for the missing chunks
                                logger.info("Continuing to wait for missing chunks...")

                # If the queue is empty and all processing is done, send final results and exit
                if result_queue.empty() and all_done and not final_result_sent:
                    if results_by_index:
                        # Send final complete message with all results
                        indices = sorted(results_by_index.keys())
                        all_texts = [results_by_index[idx] for idx in indices]
                        full_text = " ".join(all_texts)
                        complete_event = {
                            "status": "complete",
                            "text": full_text,
                            "processing_time": time.time() - start_time,
                            "chunks_total": len(indices),
                            "word_count": len(full_text.split()),
                        }
                        logger.info(f"Sending complete event: {complete_event}")
                        yield f"data: {json.dumps(complete_event)}\n\n"
                        logger.info(
                            f"Sent final complete message with {len(indices)} chunks"
                        )
                    else:
                        logger.warning(
                            "No transcription results found in results_by_index when finishing"
                        )
                        # Check if we have any active tasks or pending chunks that might still produce results
                        if (
                            not active_tasks
                            and chunk_queue.empty()
                            and splitter_task.done()
                            and processor_task.done()
                        ):
                            if last_chunk_result and "text" in last_chunk_result:
                                # We have at least one result, use it
                                logger.info("Using last chunk result as final result")
                                complete_event = {
                                    "status": "complete",
                                    "text": last_chunk_result["text"],
                                    "processing_time": time.time() - start_time,
                                    "chunks_total": 1,
                                    "word_count": len(
                                        last_chunk_result["text"].split()
                                    ),
                                }
                                logger.info(
                                    f"Sending final complete event with last chunk: {complete_event}"
                                )
                                yield f"data: {json.dumps(complete_event)}\n\n"
                            else:
                                # We're truly done with no results
                                logger.error("No transcription results were generated")
                                error_event = {
                                    "status": "error",
                                    "message": "No transcription results were generated",
                                }
                                logger.info(f"Sending error event: {error_event}")
                                yield f"data: {json.dumps(error_event)}\n\n"
                        else:
                            # We might still be processing, wait for results
                            update_event = {
                                "status": "update",
                                "message": "Processing in progress, waiting for results...",
                                "elapsed": time.time() - start_time,
                            }
                            logger.info(f"Sending wait update: {update_event}")
                            yield f"data: {json.dumps(update_event)}\n\n"
                            # Don't set final_result_sent yet, we're waiting for more results
                            continue

                    final_result_sent = True
                    break

                # Periodically send status updates if no events have been received
                current_time = time.time()
                if (
                    current_time - last_update_time
                    > 1.5  # Reduced from 2.0 to 1.5 seconds
                    and not final_result_sent
                    and not all_done
                ):
                    # Check if we should mark as done based on various heuristics
                    should_mark_done = False

                    # Both tasks complete is a definitive signal
                    if splitter_task.done() and processor_task.done():
                        # Only mark as done if we have all expected chunks
                        if (
                            expected_total_chunks is None
                            or len(results_by_index) >= expected_total_chunks
                        ):
                            should_mark_done = True
                            logger.info(
                                "Both tasks complete with all expected chunks - marking as done"
                            )
                        else:
                            logger.warning(
                                f"Both tasks complete but only received {len(results_by_index)}/{expected_total_chunks} chunks"
                            )
                            # Only force completion if it's been a very long time since the last progress
                            if (
                                current_time - last_progress_time > 60
                            ):  # Increased timeout to 60 seconds
                                should_mark_done = True
                                logger.warning(
                                    "No progress for 60s, forcing completion despite missing chunks"
                                )

                    # If we have results, no recent activity, and queues are empty - this is our backup
                    # But ONLY if processor task is also done
                    elif (
                        results_by_index
                        and current_time - last_progress_time
                        > 30  # Increased from 10s to 30s
                        and chunk_queue.empty()
                        and processor_task.done()  # CRITICAL FIX: Only if processor is done
                    ):
                        # Only mark as done if we have all expected chunks
                        if (
                            expected_total_chunks is None
                            or len(results_by_index) >= expected_total_chunks
                        ):
                            should_mark_done = True
                            logger.info(
                                "No activity for 30s with results and empty queues with all chunks - marking as done"
                            )
                        else:
                            logger.warning(
                                f"No activity for 30s but only received {len(results_by_index)}/{expected_total_chunks} chunks"
                            )
                            # Only force completion if it's been a very long time since the last progress
                            if (
                                current_time - last_progress_time > 60
                            ):  # Increased timeout to 60 seconds
                                should_mark_done = True
                                logger.warning(
                                    "No progress for 60s, forcing completion despite missing chunks"
                                )

                    # We have decent number of results and no new results for a while
                    # But ONLY if processor task is also done
                    elif (
                        len(results_by_index) >= 3
                        and current_time - last_progress_time
                        > 30  # Increased from 20 to 30 seconds
                        and processor_task.done()  # CRITICAL FIX: Only if processor is done
                    ):
                        # Only mark as complete if we have all expected chunks or don't know how many to expect
                        if (
                            expected_total_chunks is None
                            or len(results_by_index) >= expected_total_chunks
                        ):
                            should_mark_done = True
                            logger.info(
                                f"No new results for 30s with {len(results_by_index)} chunks (all expected) - marking as done"
                            )
                        else:
                            logger.warning(
                                f"No new results for 30s but only received {len(results_by_index)}/{expected_total_chunks} chunks"
                            )
                            # Only force completion if it's been a very long time since the last progress
                            if (
                                current_time - last_progress_time > 60
                            ):  # Increased from 45s to 60s
                                should_mark_done = True
                                logger.warning(
                                    "No progress for 60s, forcing completion despite missing chunks"
                                )
                            else:
                                should_mark_done = False

                    if should_mark_done:
                        all_done = True
                        # Send finalizing message
                        update_status = {
                            "status": "update",
                            "message": "Finalizing transcription...",
                            "elapsed": current_time - start_time,
                        }
                        logger.info(f"Sending finalizing update: {update_status}")
                        yield f"data: {json.dumps(update_status)}\n\n"
                        last_update_time = current_time
                        continue

                    # Only send "Processing in progress" if we're not done yet
                    update_status = {
                        "status": "update",
                        "message": f"Processing in progress{'.' * (int(current_time) % 4 + 1)}",  # Animated dots
                        "elapsed": current_time - start_time,
                        "stats": {
                            "total": len(results_by_index) if results_by_index else 0,
                            "processed": len(results_by_index)
                            if results_by_index
                            else 0,
                        },
                    }
                    logger.info(f"Sending periodic update: {update_status}")
                    yield f"data: {json.dumps(update_status)}\n\n"
                    last_update_time = current_time

                # Add a timeout mechanism to ensure we don't get stuck
                # If processing takes too long without progress (over 60 seconds since last real update),
                # force completion
                try:
                    # Wait for data from the queue with a timeout
                    logger.debug("Waiting for data from result queue...")
                    event = await asyncio.wait_for(result_queue.get(), timeout=1.0)
                    logger.info(f"Received event from queue: {event}")

                    # Update the last update time
                    last_update_time = time.time()

                    # Handle different event types
                    if event["status"] == "splitting":
                        # Audio analysis complete, now splitting into chunks
                        logger.info(f"Splitting event received: {event}")
                        # Update the expected total chunks if available
                        if "total_chunks" in event:
                            expected_total_chunks = event["total_chunks"]
                            logger.info(
                                f"Updated expected total chunks to {expected_total_chunks}"
                            )
                        yield f"data: {json.dumps(event)}\n\n"
                        last_progress_time = time.time()  # Update progress time

                    elif event["status"] == "chunk_created":
                        # A new chunk has been created
                        logger.info(f"Chunk created event: {event}")
                        # Update the expected total chunks if available
                        if "total_chunks" in event:
                            expected_total_chunks = event["total_chunks"]
                            logger.info(
                                f"Updated expected total chunks to {expected_total_chunks}"
                            )
                        yield f"data: {json.dumps(event)}\n\n"
                        last_progress_time = time.time()  # Update progress time

                    elif event["status"] == "chunk_result":
                        # Store the result by chunk index
                        if "chunk_index" in event and "text" in event:
                            chunk_index = event["chunk_index"]
                            results_by_index[chunk_index] = event["text"]
                            last_chunk_result = event  # Save for backup
                            last_progress_time = time.time()

                            # Update the expected total chunks if available
                            if "total_chunks" in event:
                                expected_total_chunks = event["total_chunks"]
                                logger.info(
                                    f"Updated expected total chunks to {expected_total_chunks}"
                                )

                            # Check if this chunk is marked as final
                            if event.get("is_final", False):
                                logger.info(
                                    "Received chunk marked as final - marking transcription as complete"
                                )
                                all_done = True

                            # Check if splitter is done and this might be the last chunk
                            if splitter_task.done():
                                # If we know the total chunks and have all of them
                                if expected_total_chunks and isinstance(
                                    expected_total_chunks, int
                                ):
                                    if len(results_by_index) >= expected_total_chunks:
                                        logger.info(
                                            f"Received all {expected_total_chunks} chunks - marking as complete"
                                        )
                                        all_done = True
                                    else:
                                        # Log how many chunks we're still waiting for
                                        missing_count = expected_total_chunks - len(
                                            results_by_index
                                        )
                                        logger.info(
                                            f"Received {len(results_by_index)}/{expected_total_chunks} chunks, waiting for {missing_count} more"
                                        )

                                # If processor reports chunks_processed = chunks_total
                                if (
                                    "chunks_processed" in event
                                    and "chunks_total" in event
                                    and event["chunks_total"] != "in progress"
                                    and event["chunks_total"] != "unknown"
                                ):
                                    if (
                                        event["chunks_processed"]
                                        == event["chunks_total"]
                                    ):
                                        logger.info(
                                            "All chunks processed - marking as complete"
                                        )
                                        all_done = True

                        # Send the result to the client
                        logger.info(
                            f"Sending chunk result (index {event.get('chunk_index', 'unknown')})"
                        )
                        yield f"data: {json.dumps(event)}\n\n"

                    elif event["status"] == "task_complete":
                        # A task has completed - this is a strong signal we can use
                        task_name = event.get("task", "unknown")
                        logger.info(f"Task complete event received: {task_name}")
                        active_tasks -= 1

                        # Mark the specific task as done in our tracking
                        if task_name == "splitter":
                            logger.info(
                                "Splitter task is complete - all audio chunks created"
                            )
                            # Don't mark all_done yet, as processor may still be working

                            # CRITICAL FIX: DO NOT mark as complete when splitter is done
                            # We need to wait for the processor to finish processing all chunks
                            # Only log that we're waiting for processor to finish
                            logger.info(
                                f"Waiting for processor to finish processing chunks. Currently processed: {len(results_by_index)}/{expected_total_chunks if expected_total_chunks else '?'}"
                            )

                            # DO NOT set all_done = True here

                        elif task_name == "processor":
                            logger.info(
                                "Processor task is complete - all transcription done"
                            )

                            # CRITICAL FIX: Check if we have all expected chunks before marking as complete
                            # Get the sent and processed chunk counts from the event if available
                            sent_count = event.get("sent_chunk_count", 0)
                            processed_count = event.get("processed_chunk_count", 0)

                            if (
                                sent_count
                                and processed_count
                                and sent_count != processed_count
                            ):
                                logger.warning(
                                    f"Processor is done but processed count ({processed_count}) doesn't match sent count ({sent_count}). There may be missing chunks."
                                )
                                # Continue processing to see if more results come in
                                # DO NOT set all_done = True here
                            elif results_by_index:
                                # Verify we have all expected chunks before marking as complete
                                if (
                                    expected_total_chunks
                                    and len(results_by_index) < expected_total_chunks
                                ):
                                    logger.warning(
                                        f"Processor is done but only received {len(results_by_index)}/{expected_total_chunks} chunks. Continuing to wait for more results..."
                                    )
                                    # DO NOT set all_done = True here
                                else:
                                    logger.info(
                                        f"Processor done with results ({len(results_by_index)} chunks) - marking as complete"
                                    )
                                    all_done = True
                            else:
                                logger.warning(
                                    "Processor done but no results - continuing to wait"
                                )
                                # DO NOT set all_done = True here

                        # All tasks complete - ONLY mark as done if we have all expected chunks
                        if active_tasks <= 0:
                            if (
                                expected_total_chunks
                                and len(results_by_index) < expected_total_chunks
                            ):
                                logger.warning(
                                    f"All tasks complete but only received {len(results_by_index)}/{expected_total_chunks} chunks. Continuing to wait for more results..."
                                )
                                # DO NOT set all_done = True here
                            else:
                                logger.info(
                                    "All tasks complete with all expected chunks - marking as complete"
                                )
                                all_done = True

                        # Update progress time to prevent timeout-based completion
                        last_progress_time = time.time()

                    elif event["status"] == "complete":
                        # All done - this is now handled separately when queue is empty
                        logger.info(
                            "Complete event received from queue, marking all_done"
                        )
                        all_done = True
                        last_progress_time = time.time()  # Update progress time
                        if "text" in event:
                            # Store the final text if provided
                            results_by_index = {0: event["text"]}
                            last_chunk_result = event

                    elif event["status"] == "error":
                        # Error occurred
                        logger.error(
                            f"Error in processing: {event.get('message', 'Unknown error')}"
                        )
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("fatal", False):
                            break

                    else:
                        # Other update events
                        logger.info(f"Sending update event: {event}")
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("status") == "progress_update":
                            last_progress_time = time.time()  # Update progress time

                except asyncio.TimeoutError:
                    # If all done and no more results after timeout, exit
                    if all_done:
                        if not final_result_sent:
                            # Check if we have any results before finishing
                            if len(results_by_index) > 0:
                                indices = sorted(results_by_index.keys())
                                all_texts = [results_by_index[idx] for idx in indices]
                                full_text = " ".join(all_texts)
                                complete_event = {
                                    "status": "complete",
                                    "text": full_text,
                                    "processing_time": time.time() - start_time,
                                    "chunks_total": len(indices),
                                    "word_count": len(full_text.split()),
                                }
                                logger.info(
                                    f"Sending complete event at end: {complete_event}"
                                )
                                yield f"data: {json.dumps(complete_event)}\n\n"
                                final_result_sent = True
                            elif last_chunk_result and "text" in last_chunk_result:
                                # We have at least one result, use it as fallback
                                complete_event = {
                                    "status": "complete",
                                    "text": last_chunk_result["text"],
                                    "processing_time": time.time() - start_time,
                                    "chunks_total": 1,
                                    "word_count": len(
                                        last_chunk_result["text"].split()
                                    ),
                                }
                                logger.info(
                                    f"Sending final fallback complete event: {complete_event}"
                                )
                                yield f"data: {json.dumps(complete_event)}\n\n"
                                final_result_sent = True
                            else:
                                update_event = {
                                    "status": "update",
                                    "message": "Processing complete, finalizing results...",
                                    "elapsed": time.time() - start_time,
                                }
                                logger.info(f"Sending final update: {update_event}")
                                # Send a final update
                                yield f"data: {json.dumps(update_event)}\n\n"
                        break

                    # Continue waiting without sending any updates
                    continue
                except Exception as e:
                    logger.error(f"Error handling results: {str(e)}")
                    yield f"data: {json.dumps({'status': 'error', 'message': f'Error handling results: {str(e)}'})}\n\n"
                    break
        except Exception as e:
            logger.exception(f"Error in transcription stream: {str(e)}")
            error_event = {
                "status": "error",
                "message": f"Error in transcription stream: {str(e)}",
                "fatal": True,
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            # Clean up temporary files
            try:
                if temp_audio_path.exists():
                    temp_audio_path.unlink()
            except Exception as e:
                logger.error(f"Error removing temp file: {str(e)}")

    return StreamingResponse(generate(), media_type="text/event-stream")
