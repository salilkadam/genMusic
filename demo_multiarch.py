#!/usr/bin/env python3
"""
Multi-Architecture Music Generation API Demo
Demonstrates the auto-detection and deployment capabilities.
"""

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

from scripts.detect_system import SystemDetector

def demo_detection():
    """Demonstrate system detection capabilities."""
    print("🎯 Multi-Architecture Music Generation API Demo")
    print("=" * 60)
    
    detector = SystemDetector()
    config = detector.get_optimal_config()
    
    print("\n🔍 System Detection Results:")
    print(f"Architecture: {detector.system_info['architecture']}")
    print(f"OS: {detector.system_info['os']}")
    print(f"CPU Cores: {detector.system_info['cpu_cores']}")
    print(f"Memory: {detector.system_info['memory']['total_gb']:.1f} GB")
    
    gpu = detector.system_info['gpu']
    if gpu['available']:
        print(f"GPU: ✅ {gpu['type'].upper()} ({gpu['count']} devices, {gpu['memory']} MB)")
    else:
        print("GPU: ❌ CPU Only")
    
    print(f"\n🎯 Optimal Configuration:")
    print(f"Dockerfile: {config['dockerfile']}")
    print(f"Service: {config['service_name']}")
    print(f"Model: {config['recommended_model']}")
    print(f"Workers: {config['max_workers']}")
    print(f"Memory Limit: {config['memory_limit']}")
    print(f"Timeout Multiplier: {config['timeout_multiplier']:.1f}x")
    
    return detector, config

def demo_architecture_support():
    """Demonstrate support for different architectures."""
    print("\n🏗️ Architecture Support Matrix:")
    print("-" * 60)
    
    architectures = [
        {
            "name": "Apple Silicon (ARM64)",
            "dockerfile": "Dockerfile.apple-silicon",
            "optimization": "ARM64 native, unified memory",
            "model": "small",
            "performance": "CPU-optimized, 2-3 min loading"
        },
        {
            "name": "Intel/AMD x86_64",
            "dockerfile": "Dockerfile.intel",
            "optimization": "Intel MKL, multi-threading",
            "model": "small/medium",
            "performance": "CPU-optimized, 3-4 min loading"
        },
        {
            "name": "NVIDIA CUDA",
            "dockerfile": "Dockerfile.cuda",
            "optimization": "GPU acceleration, CUDA 11.8",
            "model": "large",
            "performance": "GPU-accelerated, 30s loading"
        }
    ]
    
    for arch in architectures:
        print(f"\n{arch['name']}:")
        print(f"  Dockerfile: {arch['dockerfile']}")
        print(f"  Optimization: {arch['optimization']}")
        print(f"  Recommended Model: {arch['model']}")
        print(f"  Performance: {arch['performance']}")

def demo_deployment_options():
    """Demonstrate deployment options."""
    print("\n🚀 Deployment Options:")
    print("-" * 60)
    
    options = [
        {
            "command": "python3 deploy.py --detect-only",
            "description": "Detect system configuration without deploying"
        },
        {
            "command": "python3 deploy.py",
            "description": "Auto-detect and deploy optimal configuration"
        },
        {
            "command": "python3 deploy.py --force-rebuild",
            "description": "Force rebuild Docker images"
        },
        {
            "command": "python3 deploy.py --tensordock",
            "description": "Generate TensorDock GPU deployment"
        },
        {
            "command": "python3 deploy.py --save-config",
            "description": "Save system configuration to JSON"
        }
    ]
    
    for option in options:
        print(f"\n{option['command']}")
        print(f"  {option['description']}")

def demo_performance_expectations():
    """Show performance expectations for different configurations."""
    print("\n📊 Performance Expectations:")
    print("-" * 60)
    
    configs = [
        {
            "system": "Apple Silicon (M2 Pro, 32GB)",
            "model": "small",
            "loading": "2-3 minutes",
            "generation": "30 seconds"
        },
        {
            "system": "Intel x86_64 (8+ cores, 16GB)",
            "model": "small",
            "loading": "3-4 minutes",
            "generation": "45 seconds"
        },
        {
            "system": "NVIDIA GPU (RTX 4090, 24GB)",
            "model": "large",
            "loading": "30 seconds",
            "generation": "15 seconds"
        }
    ]
    
    for config in configs:
        print(f"\n{config['system']}:")
        print(f"  Model: {config['model']}")
        print(f"  Loading: {config['loading']}")
        print(f"  Generation: {config['generation']}")

def demo_api_usage():
    """Show API usage examples."""
    print("\n🎵 API Usage Examples:")
    print("-" * 60)
    
    examples = [
        {
            "title": "Create Music Generation Task",
            "command": '''curl -X POST "http://localhost:8000/generate-music/" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "upbeat jazz melody", "duration": 15, "model_size": "small"}'
'''
        },
        {
            "title": "Check Task Status",
            "command": "curl -X GET \"http://localhost:8000/task/{task_id}/status\""
        },
        {
            "title": "Download Generated Audio",
            "command": "curl -X GET \"http://localhost:8000/task/{task_id}/download/song1\" -o music.wav"
        },
        {
            "title": "List All Tasks",
            "command": "curl -X GET \"http://localhost:8000/tasks\""
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print(f"  {example['command']}")

def demo_tensordock_setup():
    """Show TensorDock setup process."""
    print("\n🌐 TensorDock Deployment Process:")
    print("-" * 60)
    
    steps = [
        "1. Generate TensorDock configuration: python3 deploy.py --tensordock",
        "2. Upload files to TensorDock instance",
        "3. Run deployment script: ./deploy_tensordock.sh",
        "4. API available at: http://your-instance-ip:8000"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print("\n  TensorDock Optimizations:")
    print("  - CUDA 11.8 with GPU acceleration")
    print("  - Large model support (8.7GB)")
    print("  - 16GB memory limit")
    print("  - Persistent model caching")

def main():
    """Main demonstration function."""
    detector, config = demo_detection()
    demo_architecture_support()
    demo_deployment_options()
    demo_performance_expectations()
    demo_api_usage()
    demo_tensordock_setup()
    
    print("\n✅ Multi-Architecture System Ready!")
    print("=" * 60)
    print(f"Your system is configured for: {config['dockerfile']}")
    print(f"Recommended model: {config['recommended_model']}")
    print(f"Service name: {config['service_name']}")
    print("\nTo deploy:")
    print("  python3 deploy.py")
    print("\nTo generate TensorDock config:")
    print("  python3 deploy.py --tensordock")
    print("\nFor more information, see README_MultiArch.md")

if __name__ == "__main__":
    main() 