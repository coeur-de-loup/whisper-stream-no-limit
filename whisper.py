import asyncio

import os
import time
import httpx
from typing import Optional

from logger import logger

# Strip whitespace from API key to prevent header issues
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"

# Print the first and last few characters for debugging (safe to log)
key_preview = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}"
logger.info(f"Using API key (preview): {key_preview}, length: {len(OPENAI_API_KEY)}")

WHISPER_API_URL = os.getenv(
    "WHISPER_API_URL", "https://api.openai.com/v1/audio/transcriptions"
)


async def transcribe(
    chunk_path: str, language: Optional[str] = None, prompt: Optional[str] = None
) -> str:
    """Send audio chunk to OpenAI Whisper API for transcription"""
    chunk_name = os.path.basename(chunk_path)
    logger.info(f"Sending chunk {chunk_name} to Whisper API")

    # Explicitly format the authorization header with no unexpected characters
    auth_header = f"Bearer {OPENAI_API_KEY}"
    headers = {"Authorization": auth_header}

    form_data = {"model": "whisper-1", "response_format": "text"}
    start_time = time.time()  # Define start_time here so it's always available

    if language:
        form_data["language"] = language

    if prompt:
        form_data["prompt"] = prompt

    try:
        # Open the file when needed to avoid keeping it open for long periods
        files = {
            "file": (os.path.basename(chunk_path), open(chunk_path, "rb"), "audio/mpeg")
        }

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                WHISPER_API_URL, headers=headers, data=form_data, files=files
            )

        # Check for successful response
        if response.status_code != 200:
            error_msg = f"Whisper API error: {response.text}"
            logger.error(f"Error for chunk {chunk_name}: {error_msg}")
            raise Exception(error_msg)

        # Log processing time
        processing_time = time.time() - start_time
        logger.info(
            f"Received response for chunk {chunk_name} in {processing_time:.2f}s"
        )

        return response.text

    except httpx.TimeoutException:
        logger.error(
            f"Timeout for chunk {chunk_name} after {time.time() - start_time:.2f}s"
        )
        raise Exception(f"Request timeout for chunk {chunk_name}")
    except Exception as e:
        logger.exception(f"Error transcribing chunk {chunk_path}: {str(e)}")

        # For debugging purposes during development, return a placeholder
        # Remove this in production:
        if not OPENAI_API_KEY:
            logger.warning("No API key set, returning placeholder text")
            await asyncio.sleep(2)  # Simulate processing time
            return f"Placeholder text for {chunk_name} - API key not configured"

        raise
    finally:
        # Clean up the file after processing
        try:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
                logger.debug(f"Removed chunk file: {chunk_path}")
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up chunk file {chunk_path}: {cleanup_error}")
