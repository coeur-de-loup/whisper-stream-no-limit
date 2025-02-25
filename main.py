from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import health, transcribe, stream_transcription, options

# Initialize FastAPI app
app = FastAPI(title="Whisper Transcription API")

# Add CORS middleware to allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include all route modules
app.include_router(health.router)
app.include_router(transcribe.router)
app.include_router(stream_transcription.router)
app.include_router(options.router)

# If this file is run directly, start the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=2)
