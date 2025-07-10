#!/bin/bash
# TensorDock Deployment Script
set -e

echo "🎯 TensorDock Music Generation API Deployment"
echo "=============================================="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected. This deployment requires CUDA."
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

# Create model directory
mkdir -p ./models
mkdir -p ./output

# Build and deploy
echo "🏗️ Building CUDA-optimized container..."
docker-compose -f docker-compose.tensordock.yml up --build -d

echo "⏳ Waiting for service to start..."
sleep 30

# Health check
echo "🏥 Checking service health..."
curl -f http://localhost:8000/health || {
    echo "❌ Health check failed. Checking logs..."
    docker-compose -f docker-compose.tensordock.yml logs
    exit 1
}

echo "✅ TensorDock deployment completed!"
echo "🌐 API available at: http://localhost:8000"
echo "📋 Health: http://localhost:8000/health"
echo "📖 Docs: http://localhost:8000/docs"
echo "🎮 GPU Status: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
