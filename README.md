# GenMusic - Multi-Architecture Music Generation API

[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://docker.com)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-red.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready, multi-architecture FastAPI music generation server powered by Facebook's MusicGen models. Features intelligent system detection, auto-optimized Docker deployments, and seamless support for Apple Silicon, Intel/AMD x86_64, and NVIDIA CUDA environments.

## 🚀 Key Features

- **🎵 AI Music Generation**: Powered by Facebook's MusicGen models (Small, Medium, Large)
- **🔍 Smart Detection**: Automatically detects system architecture and capabilities
- **🏗️ Multi-Architecture**: Native support for Apple Silicon, Intel/AMD, and NVIDIA CUDA
- **🐳 Docker Optimized**: Architecture-specific Dockerfiles with performance optimizations
- **☁️ Cloud Ready**: Special TensorDock configuration for GPU cloud deployment
- **⚡ Async Processing**: Non-blocking API with task tracking and progress monitoring
- **📊 Production Ready**: Built-in health checks, monitoring, and error handling
- **📁 File Management**: Easy download of generated audio files

## 🏗️ Architecture Support

### Apple Silicon (ARM64)
- **File**: `Dockerfile.apple-silicon`
- **Optimization**: ARM64 native builds, unified memory architecture
- **Performance**: ~2-3 minutes model loading, ~30s generation

### Intel/AMD x86_64
- **File**: `Dockerfile.intel`
- **Optimization**: Intel MKL, multi-threading, CPU optimization
- **Performance**: ~3-4 minutes model loading, ~45s generation

### NVIDIA CUDA (GPU)
- **File**: `Dockerfile.cuda`
- **Optimization**: CUDA 11.8, GPU acceleration, large model support
- **Performance**: ~30s model loading, ~15s generation

## 🚀 Quick Start

### 1. Auto-Detection and Deployment
```bash
# Clone the repository
git clone https://github.com/salilkadam/genMusic.git
cd genMusic

# Auto-detect system and deploy
python3 deploy.py

# Or detect without deploying
python3 deploy.py --detect-only
```

### 2. Manual Architecture Selection
```bash
# Force specific architecture
docker-compose -f docker-compose.yml up --build -d

# Use TensorDock configuration
docker-compose -f docker-compose.tensordock.yml up --build -d
```

### 3. TensorDock Cloud Deployment
```bash
# Generate TensorDock configuration
python3 deploy.py --tensordock

# Deploy on TensorDock instance
./deploy_tensordock.sh
```

## 📋 API Usage

### Task-Based Music Generation
```bash
# Create a music generation task
curl -X POST "http://localhost:8000/generate-music/" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "upbeat jazz piano melody", 
    "duration": 15, 
    "model_size": "small"
  }'

# Response: {"task_id": "uuid-string", "status": "pending", ...}
```

### Monitor Progress
```bash
# Check task status
curl -X GET "http://localhost:8000/task/{task_id}/status"

# List all tasks
curl -X GET "http://localhost:8000/tasks"

# Get system health
curl -X GET "http://localhost:8000/health"
```

### Download Generated Audio
```bash
# List available files
curl -X GET "http://localhost:8000/task/{task_id}/files"

# Download audio file
curl -X GET "http://localhost:8000/task/{task_id}/download/song1" -o music.wav
```

## 🛠️ System Requirements

### Minimum Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for models and containers
- **Docker**: 20.10+ with Docker Compose

### GPU Requirements (Optional)
- **NVIDIA GPU**: CUDA 11.8+ compatible
- **VRAM**: 8GB+ for large models, 4GB+ for medium
- **Driver**: NVIDIA Driver 450.80+

## 📊 Model Options

| Model | Size | Quality | CPU Performance | GPU Performance |
|-------|------|---------|----------------|-----------------|
| Small | ~2.2GB | Good | ✅ Recommended | ⚡ Very Fast |
| Medium | ~4GB | Better | ⚠️ Slow | ✅ Fast |
| Large | ~8.7GB | Best | ❌ Not Recommended | ✅ Recommended |

## 🔧 Project Structure

```
genMusic/
├── 📁 Core Application
│   ├── music.py                    # Original FastAPI server
│   ├── music_optimized.py          # Enhanced with task tracking
│   └── requirements.txt            # Python dependencies
│
├── 🐳 Docker Configurations
│   ├── Dockerfile.apple-silicon    # ARM64 optimized
│   ├── Dockerfile.intel           # x86_64 optimized
│   ├── Dockerfile.cuda            # NVIDIA CUDA optimized
│   ├── docker-compose.yml         # Auto-generated configuration
│   └── docker-compose.tensordock.yml # TensorDock deployment
│
├── 🤖 Smart Deployment
│   ├── deploy.py                  # Intelligent deployment system
│   ├── scripts/detect_system.py   # System detection module
│   └── deploy_tensordock.sh       # TensorDock deployment script
│
├── 📖 Documentation
│   ├── README.md                  # This file
│   ├── README_MultiArch.md        # Detailed architecture guide
│   └── demo_multiarch.py          # Interactive demonstration
│
└── 📁 Runtime Directories
    ├── models/                    # Model cache directory
    ├── output/                    # Generated audio files
    └── scripts/                   # Utility scripts
```

## 🎯 Deployment Options

### Option 1: Auto-Detection (Recommended)
```bash
python3 deploy.py
```
Automatically detects your system and deploys the optimal configuration.

### Option 2: Manual Configuration
```bash
# Apple Silicon
docker-compose -f docker-compose.yml up -d

# Intel/AMD
docker-compose -f docker-compose.yml up -d

# NVIDIA GPU
docker-compose -f docker-compose.tensordock.yml up -d
```

### Option 3: TensorDock Cloud
```bash
# Generate TensorDock configuration
python3 deploy.py --tensordock

# Upload to TensorDock and run
./deploy_tensordock.sh
```

## 🌐 TensorDock Deployment

Perfect for GPU-accelerated music generation in the cloud:

1. **Generate Configuration**:
   ```bash
   python3 deploy.py --tensordock
   ```

2. **Upload Files**: Transfer all files to your TensorDock instance

3. **Deploy**:
   ```bash
   ./deploy_tensordock.sh
   ```

4. **Access**: API available at `http://your-instance-ip:8000`

### TensorDock Benefits
- **GPU Acceleration**: 10x faster than CPU processing
- **Large Model Support**: Handle 8.7GB models efficiently
- **Scalable**: Multiple GPU support
- **Cost Effective**: Pay per use

## 🔍 System Detection

The system automatically detects:
- **Architecture**: ARM64, x86_64, etc.
- **GPU**: NVIDIA CUDA, AMD ROCm
- **Memory**: Available RAM
- **Docker**: Version and capabilities

```bash
# View detection results
python3 deploy.py --detect-only

# Save configuration
python3 deploy.py --save-config
```

## 📈 Performance Optimization

### CPU Optimization
- **Thread Management**: Optimized for multi-core processors
- **Memory Management**: Efficient model caching
- **Intel MKL**: Accelerated math operations on Intel CPUs

### GPU Optimization
- **CUDA Acceleration**: Native GPU processing
- **Memory Management**: Efficient VRAM usage
- **Model Parallel**: Large model support

### Docker Optimization
- **Layer Caching**: Faster rebuilds
- **Multi-stage Builds**: Smaller image sizes
- **Resource Limits**: Prevents system overload

## 🚨 Troubleshooting

### Common Issues

#### Docker Build Fails
```bash
# Clean up and rebuild
docker system prune -a
python3 deploy.py --force-rebuild
```

#### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

#### Model Loading Timeout
```bash
# Check system resources
python3 deploy.py --detect-only

# Use smaller model
# Change "large" to "small" in requests
```

#### Out of Memory
```bash
# Increase Docker memory limit
# Edit docker-compose.yml: mem_limit: 8g

# Use smaller model
# Recommend "small" for systems with <16GB RAM
```

## 🔧 Advanced Configuration

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

### Custom Model Configuration
Edit `music_optimized.py` to add custom models:
```python
MODEL_CONFIGS = {
    "custom": {
        "model_name": "facebook/musicgen-custom",
        "description": "Custom model",
        "size_gb": 5.0
    }
}
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup
```bash
# Clone repository
git clone https://github.com/salilkadam/genMusic.git
cd genMusic

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run local development server
python music_optimized.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Facebook MusicGen**: For the incredible music generation models
- **Hugging Face**: For the transformers library and model hosting
- **FastAPI**: For the high-performance API framework
- **Docker**: For containerization and multi-architecture support
- **TensorDock**: For affordable GPU cloud infrastructure

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/salilkadam/genMusic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/salilkadam/genMusic/discussions)
- **Email**: salil.kadam@gmail.com

## 🔗 Links

- **Demo**: [Live Demo](https://demo.genmusic.ai) (Coming Soon)
- **Documentation**: [Full Documentation](https://docs.genmusic.ai) (Coming Soon)
- **Docker Hub**: [Docker Images](https://hub.docker.com/r/salilkadam/genmusic) (Coming Soon)

---

**Made with ❤️ by [Salil Kadam](https://github.com/salilkadam)** 