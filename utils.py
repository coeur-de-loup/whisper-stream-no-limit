import logging
import os
from pathlib import Path
import math
import subprocess
import asyncio
import uuid


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Get logger
logger = logging.getLogger("whisper-api")
logger.setLevel(logging.INFO)


async def cleanup_temp_files(session_dir: Path, original_file: str):
    """Clean up temporary files after processing"""
    try:
        # Remove the original uploaded file
        logger.info(f"Cleaning up temporary file: {original_file}")
        os.remove(original_file)
    except Exception as e:
        logger.warning(f"Failed to remove original file: {str(e)}")

    try:
        # Remove all files in the session directory
        logger.info(f"Cleaning up session directory: {session_dir}")
        for file_path in session_dir.glob("*"):
            try:
                os.remove(file_path)
                logger.debug(f"Removed chunk file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove chunk file {file_path}: {str(e)}")

        # Remove the session directory
        session_dir.rmdir()
        logger.info(f"Removed session directory: {session_dir}")
    except Exception as e:
        logger.warning(f"Failed to fully clean up session directory: {str(e)}")


# Function to split audio for streaming mode
async def split_audio(audio_path, chunk_queue, result_queue):
    """Split audio file into chunks and put them in the queue for processing"""
    # Initialize variables used in finally block
    created_chunks = 0
    chunk_order = []

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
            str(audio_path),
        ]

        logger.info("Getting audio duration")
        await result_queue.put(
            {"status": "update", "message": "Analyzing audio file..."}
        )

        duration_process = subprocess.run(duration_cmd, capture_output=True, text=True)
        if duration_process.returncode != 0:
            logger.error("Failed to get audio duration")
            await result_queue.put(
                {
                    "status": "error",
                    "message": "Error getting audio duration",
                    "fatal": True,
                }
            )
            return

        total_duration = float(duration_process.stdout.strip())
        logger.info(f"Audio duration: {total_duration:.2f} seconds")

        # Calculate optimal chunk duration (around 30 seconds per chunk)
        chunk_duration = min(
            60, max(30, total_duration / 20)
        )  # At least 20 chunks or 30-60s each
        chunk_count = math.ceil(total_duration / chunk_duration)

        logger.info(
            f"Splitting into approximately {chunk_count} chunks of {chunk_duration:.2f}s each"
        )

        # Send splitting event
        splitting_event = {
            "status": "splitting",
            "total_chunks": chunk_count,
            "duration": total_duration,
            "message": "Starting to split audio into chunks",
        }
        logger.info(f"Sending splitting event: {splitting_event}")
        await result_queue.put(splitting_event)

        # Create temp directory for chunks if it doesn't exist
        chunks_dir = Path("temp/chunks")
        chunks_dir.mkdir(exist_ok=True, parents=True)

        # Allow time for event to be processed
        await asyncio.sleep(0.1)

        # Send first progress update for chunking only
        chunking_progress = {
            "status": "progress_update",
            "message": "Starting to create audio chunks",
            "chunking_progress": {"created": 0, "total": chunk_count},
            "transcription_progress": {"processed": 0, "total": chunk_count},
        }
        await result_queue.put(chunking_progress)

        # Create chunks
        created_chunks = 0

        for i in range(chunk_count):
            start_time_sec = i * chunk_duration
            duration_sec = min(chunk_duration, total_duration - start_time_sec)

            if duration_sec <= 0:
                break

            # Send update about creating this chunk
            chunk_update = {
                "status": "chunk_created",
                "chunk": i + 1,
                "total_chunks": chunk_count,
                "message": f"Creating chunk {i + 1}/{chunk_count}",
            }
            logger.info(f"Sending chunk creation update: {chunk_update}")
            await result_queue.put(chunk_update)

            # Allow time for event to be processed
            await asyncio.sleep(0.05)

            # Create a unique filename for this chunk
            output_path = chunks_dir / f"chunk_{i:03d}_{uuid.uuid4()}.mp3"

            # Use ffmpeg to extract the chunk
            cmd = [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-ss",
                str(start_time_sec),
                "-t",
                str(duration_sec),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                "-y",  # Overwrite output files
                str(output_path),
            ]

            # Use asyncio to run the command non-blocking
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            # Wait for the process to complete
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"Error creating chunk {i + 1}: {stderr.decode()}")
                await result_queue.put(
                    {
                        "status": "error",
                        "message": f"Error creating chunk {i + 1}: {stderr.decode()}",
                    }
                )
                continue

            # Increment the created chunks counter
            created_chunks += 1

            # Put the chunk in the queue for processing
            logger.info(f"Created chunk {i + 1}/{chunk_count}: {output_path}")
            await chunk_queue.put((i, str(output_path)))
            chunk_order.append(i)  # Track the chunk order

            # Send progress update for chunking
            if (i + 1) % 2 == 0 or i == 0 or i == chunk_count - 1:
                chunking_progress = {
                    "status": "progress_update",
                    "message": f"Created {i + 1}/{chunk_count} chunks",
                    "chunking_progress": {"created": i + 1, "total": chunk_count},
                    "transcription_progress": {"processed": 0, "total": chunk_count},
                }
                await result_queue.put(chunking_progress)

        # Signal that all chunks have been created
        logger.info(f"All {len(chunk_order)} chunks created and queued for processing")

        # CRITICAL FIX: Send a definitive count of created chunks to help the processor track completion
        created_chunks_event = {
            "status": "chunks_created_count",
            "created_chunks": created_chunks,
            "chunk_count": chunk_count,
            "chunk_order_length": len(chunk_order),
        }
        logger.info(f"Sending created chunks count: {created_chunks_event}")
        await result_queue.put(created_chunks_event)

        # Final chunking progress update
        final_chunking_progress = {
            "status": "progress_update",
            "message": f"All {chunk_count} chunks created, now transcribing",
            "chunking_progress": {"created": chunk_count, "total": chunk_count},
            "transcription_progress": {"processed": 0, "total": chunk_count},
            "chunk_order_length": len(
                chunk_order
            ),  # Add chunk order length for debugging
            "actual_created_chunks": created_chunks,  # Add actual created chunks count
        }
        await result_queue.put(final_chunking_progress)

    except Exception as e:
        logger.exception(f"Error in split_audio: {str(e)}")
        await result_queue.put(
            {
                "status": "error",
                "message": f"Error splitting audio: {str(e)}",
                "fatal": True,
            }
        )
    finally:
        # Signal that this task is complete
        await result_queue.put(
            {
                "status": "task_complete",
                "task": "splitter",
                "created_chunks": created_chunks,
                "chunk_order_length": len(chunk_order),
            }
        )
