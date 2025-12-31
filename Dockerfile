# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch first with CPU index URL
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Create upload directory
RUN mkdir -p temp_uploads

# Expose port
EXPOSE $PORT

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --keep-alive 5 app:app