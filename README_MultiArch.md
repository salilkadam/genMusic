# Multi-Architecture Music Generation API

A smart, auto-detecting Docker deployment system for the FastAPI Music Generation API that automatically optimizes for your hardware configuration.

## 🎯 Features

- **Auto-Detection**: Automatically detects your system architecture and GPU capabilities
- **Multi-Architecture Support**: Apple Silicon (ARM64), Intel/AMD x86_64, and NVIDIA CUDA
- **Smart Optimization**: Chooses optimal Docker configuration based on your hardware
- **TensorDock Ready**: Special configuration for TensorDock GPU cloud deployment
- **Task-Based Processing**: Non-blocking API with task tracking and file downloads
- **Production Ready**: Built-in health checks, monitoring, and error handling

## 🏗️ Architecture Support

### 1. Apple Silicon (ARM64)
- **Dockerfile**: `Dockerfile.apple-silicon`
- **Optimization**: ARM64 native builds, optimized for Apple M1/M2/M3 chips
- **Models**: Recommended `small` model for CPU processing
- **Memory**: Optimized for unified memory architecture

### 2. Intel/AMD x86_64
- **Dockerfile**: `Dockerfile.intel`
- **Optimization**: Intel MKL optimizations, multi-threading support
- **Models**: Recommended `small` model, can handle `medium` with sufficient RAM
- **Memory**: Configurable memory limits based on available RAM

### 3. NVIDIA CUDA (GPU)
- **Dockerfile**: `Dockerfile.cuda`
- **Optimization**: CUDA 11.8, GPU acceleration, large model support
- **Models**: Recommended `large` model for high-end GPUs, `medium` for mid-range
- **Memory**: GPU memory-aware model selection

## 🚀 Quick Start

### 1. System Detection
```bash
# Check your system configuration
python3 deploy.py --detect-only

# Save configuration to file
python3 deploy.py --save-config
```

### 2. Automatic Deployment
```bash
# Auto-detect and deploy optimal configuration
python3 deploy.py

# Force rebuild of Docker images
python3 deploy.py --force-rebuild
```

### 3. TensorDock Deployment
```bash
# Generate TensorDock-optimized configuration
python3 deploy.py --tensordock

# Run the generated script on TensorDock
./deploy_tensordock.sh
```

## 📋 System Requirements

### Minimum Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for models and containers
- **Docker**: Docker 20.10+ and Docker Compose 1.29+

### GPU Requirements (Optional)
- **NVIDIA GPU**: CUDA 11.8+ compatible
- **VRAM**: 8GB+ for large models, 4GB+ for medium models
- **Driver**: NVIDIA Driver 450.80+

## 🔧 Configuration Files

### Generated Files
- `docker-compose.yml` - Auto-generated for your architecture
- `docker-compose.tensordock.yml` - TensorDock-optimized configuration
- `deployment_config.json` - System detection and configuration data
- `deploy_tensordock.sh` - TensorDock deployment script

### Architecture-Specific Dockerfiles
- `Dockerfile.apple-silicon` - ARM64 optimized
- `Dockerfile.intel` - x86_64 optimized
- `Dockerfile.cuda` - NVIDIA CUDA optimized

## 🎵 API Usage

### Task-Based Music Generation
```bash
# Create a music generation task
curl -X POST "http://localhost:8000/generate-music/" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "upbeat jazz melody", "duration": 15, "model_size": "small"}'

# Response: {"task_id": "uuid-string", "status": "pending", ...}
```

### Check Task Status
```bash
# Check task progress
curl -X GET "http://localhost:8000/task/{task_id}/status"

# List all tasks
curl -X GET "http://localhost:8000/tasks"
```

### Download Generated Audio
```bash
# List available files
curl -X GET "http://localhost:8000/task/{task_id}/files"

# Download audio file
curl -X GET "http://localhost:8000/task/{task_id}/download/song1" -o music.wav
```

## 🛠️ Advanced Configuration

### Manual Architecture Selection
```bash
# Force Intel configuration
docker-compose -f docker-compose.yml up --build -d

# Force CUDA configuration (with GPU support)
docker-compose -f docker-compose.tensordock.yml up --build -d
```

### Environment Variables
```bash
# CPU optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all

# Model caching
export TRANSFORMERS_CACHE=/path/to/models
export HF_HOME=/path/to/models
```

### Model Size Selection
- **Small**: ~2.2GB, fast processing, good quality
- **Medium**: ~4GB, better quality, moderate processing time
- **Large**: ~8.7GB, best quality, requires GPU for reasonable performance

## 📊 Performance Expectations

### Apple Silicon (M2 Pro, 32GB RAM)
- **Small Model**: ~2-3 minutes loading, ~30s generation
- **Medium Model**: ~4-5 minutes loading, ~60s generation
- **Large Model**: Not recommended for CPU-only

### Intel/AMD x86_64 (8+ cores, 16GB RAM)
- **Small Model**: ~3-4 minutes loading, ~45s generation
- **Medium Model**: ~6-8 minutes loading, ~90s generation
- **Large Model**: Not recommended for CPU-only

### NVIDIA GPU (RTX 4090, 24GB VRAM)
- **Small Model**: ~30s loading, ~10s generation
- **Medium Model**: ~60s loading, ~15s generation
- **Large Model**: ~120s loading, ~30s generation

## 🔍 Troubleshooting

### Common Issues

#### 1. Docker Build Fails
```bash
# Check Docker version
docker --version
docker-compose --version

# Clean up and rebuild
docker system prune -a
python3 deploy.py --force-rebuild
```

#### 2. GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

#### 3. Model Loading Timeout
```bash
# Check logs
docker-compose logs -f

# Increase timeout in music_optimized.py
# timeout=600  # 10 minutes
```

#### 4. Out of Memory
```bash
# Reduce model size
# Use "small" instead of "medium" or "large"

# Increase Docker memory limit
# mem_limit: 8g
```

### Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check service status
docker-compose ps

# View logs
docker-compose logs --tail 50
```

## 🌐 TensorDock Deployment

### Prerequisites
- TensorDock account with GPU instance
- NVIDIA GPU with CUDA support
- Docker and Docker Compose installed

### Deployment Steps
1. **Generate Configuration**:
   ```bash
   python3 deploy.py --tensordock
   ```

2. **Upload Files**: Upload all files to your TensorDock instance

3. **Run Deployment**:
   ```bash
   ./deploy_tensordock.sh
   ```

4. **Access API**: The API will be available at your instance's IP on port 8000

### TensorDock Optimizations
- **GPU Memory**: Optimized for high-VRAM GPUs
- **Model Selection**: Defaults to `large` model for best quality
- **Concurrent Processing**: Supports multiple parallel generations
- **Persistent Storage**: Models cached to persistent storage

## 📈 Monitoring and Scaling

### Built-in Monitoring
- **Health Endpoint**: `/health` - System status and cached models
- **Task Status**: Real-time progress tracking
- **Performance Metrics**: Generation times and resource usage

### Scaling Options
- **Horizontal**: Run multiple instances behind a load balancer
- **Vertical**: Increase memory and CPU limits
- **GPU Scaling**: Use multiple GPUs with CUDA_VISIBLE_DEVICES

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new architectures
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Facebook MusicGen**: For the incredible music generation models
- **Hugging Face**: For the transformers library and model hosting
- **FastAPI**: For the high-performance API framework
- **Docker**: For containerization and multi-architecture support 