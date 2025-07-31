# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY .env.local* ./

# Install Python dependencies
RUN uv sync

# Activate virtual environment by setting PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose the healthcheck port
# This allows Docker and orchestration systems to check if the container is healthy
EXPOSE 8081

# Default command
CMD ["python", "src/audio3.py", "dev"]
