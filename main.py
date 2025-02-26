from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from routes import health, transcribe, stream_transcription, options
import os

# Initialize FastAPI app
app = FastAPI(title="Whisper Transcription API")

# API key settings
API_KEY = os.environ.get(
    "API_KEY", "your-default-api-key"
)  # Set a default key for development
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# Authentication dependency
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key header is missing"
        )
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key_header


# Add CORS middleware to allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include all route modules with API key authentication
# Public health check doesn't require authentication
app.include_router(health.router)

# Protected routes require API key authentication
app.include_router(transcribe.router, dependencies=[Depends(get_api_key)])
app.include_router(stream_transcription.router, dependencies=[Depends(get_api_key)])
app.include_router(options.router, dependencies=[Depends(get_api_key)])

# If this file is run directly, start the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=2)
