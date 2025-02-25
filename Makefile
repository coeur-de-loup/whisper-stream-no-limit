# Whisper Transcription Service - Development Makefile
.PHONY: help build run stop restart logs shell clean status dev dev-logs test all

# Default target when just running 'make'
help:
	@echo "Whisper Transcription Service - Available commands:"
	@echo "  make build      - Build the Docker image"
	@echo "  make run        - Run the service (doesn't rebuild)"
	@echo "  make up         - Build and run the service (same as docker compose up -d)"
	@echo "  make stop       - Stop the running service"
	@echo "  make restart    - Restart the service"
	@echo "  make logs       - View service logs"
	@echo "  make shell      - Open a shell in the running container"
	@echo "  make clean      - Remove containers and local images"
	@echo "  make status     - Check status of services"
	@echo "  make dev        - Run in development mode (with code reloading)"
	@echo "  make dev-logs   - View development mode logs"
	@echo "  make test       - Run tests"
	@echo "  make all        - Clean, build, and start service"

# Build the Docker image
build:
	docker compose build

# Run the service without rebuilding
run:
	docker compose up -d --no-build

# Build and run the service
up:
	docker compose up -d

# Stop the running service
stop:
	docker compose down

# Restart the service
restart:
	docker compose restart

# View service logs
logs:
	docker compose logs -f

# Open a shell in the running container
shell:
	docker compose exec whisper-transcription-service sh

# Remove containers and local images
clean:
	docker compose down --rmi local
	docker system prune -f

# Check status of containers
status:
	docker compose ps

# Run in development mode with code reloading (assumes Python is installed locally)
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# View development logs
dev-logs:
	tail -f logs/*.log

# Run tests (can be expanded as needed)
test:
	docker compose run --rm whisper-transcription-service pytest -xvs

# Clean everything and start fresh
all: clean build run 