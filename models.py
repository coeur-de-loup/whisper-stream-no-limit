from pydantic import BaseModel
from typing import Optional


# Models
class TranscriptionRequest(BaseModel):
    language: Optional[str] = None
    prompt: Optional[str] = None


class TranscriptionResponse(BaseModel):
    text: str
    processing_time: float


# Add streaming response model
class StreamingTranscriptionResponse(BaseModel):
    chunk_index: int
    total_chunks: int
    text: str
    is_final: bool
