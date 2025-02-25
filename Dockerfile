FROM --platform=${BUILDPLATFORM} python:3.11-slim as builder

# Set build arguments for multi-platform support
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build final image
FROM --platform=${TARGETPLATFORM} python:3.11-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MAX_WORKERS=2

# Note: Other environment variables like OPENAI_API_KEY, MAX_PARALLEL_CHUNKS, and WHISPER_API_URL
# will be passed at runtime via docker-compose or docker run

# Copy application code
COPY . /app/

# Create directory for temporary files
RUN mkdir -p /tmp/whisper_chunks && chmod 777 /tmp/whisper_chunks

# Expose port
EXPOSE 8000

# Run with multiple workers for better performance
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]