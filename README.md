# Whisper Transcription Service

A FastAPI-based service for audio transcription using OpenAI's Whisper API.

## Development Tools

This project provides two ways to manage the development workflow:

### Using Make Commands

A Makefile is included with common development tasks:

```bash
# Show all available commands
make help

# Build the Docker image
make build

# Run without rebuilding
make run

# Build and run (equivalent to docker-compose up -d)
make up

# View logs
make logs

# Open a shell in the container
make shell

# Stop the service
make stop

# Check status
make status

# And more...
```

### Using the Run Script

A convenient shell script wrapper that uses the Makefile:

```bash
# Show help
./run.sh help

# Build the Docker image
./run.sh build

# Run without rebuilding
./run.sh run

# Build and run
./run.sh up

# View logs
./run.sh logs

# And more...
```

## Development Mode

For local development with code reloading:

```bash
# Run in development mode
make dev
# or
./run.sh dev
```

## Project Structure

- `main.py` - FastAPI application entry point
- `routes/` - API route handlers
- `whisper.py` - Whisper API integration
- `models.py` - Data models
- `utils.py` - Utility functions
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-container orchestration
- `Makefile` - Development workflow automation

## Environment Variables

The service requires the following environment variables in a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
MAX_PARALLEL_CHUNKS=10
WHISPER_API_URL=https://api.openai.com/v1/audio/transcriptions
``` 