# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models

# Create models directory
RUN mkdir -p /models

# Copy application code
COPY music.py .

# Expose port
EXPOSE 8000

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /models
USER appuser

# Command to run the application
CMD ["uvicorn", "music:app", "--host", "0.0.0.0", "--port", "8000"] 