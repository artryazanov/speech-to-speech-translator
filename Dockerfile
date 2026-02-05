# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# ffmpeg is required for pydub
RUN apt-get update && apt-get install -y \
    ffmpeg \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user for security (optional but good practice)
# But for simple CLI file manipulation, running as root inside container 
# is often easier for volume permissions unless specific UID/GID handling is added.
# keeping it simple for now, but ensure output directory is writable.

# Default command
ENTRYPOINT ["python", "-m", "speech_translator.cli"]
