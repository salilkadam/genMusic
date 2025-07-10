#!/usr/bin/env python3
"""
Smart Deployment Script for Music Generation API
Auto-detects system capabilities and deploys optimal Docker configuration.
"""

import os
import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

try:
    from scripts.detect_system import SystemDetector
except ImportError:
    print("❌ Could not import SystemDetector. Please run from the project root directory.")
    sys.exit(1)

class SmartDeployer:
    def __init__(self):
        self.detector = SystemDetector()
        self.config = self.detector.get_optimal_config()
        self.project_root = Path(__file__).parent
        
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml based on detected system."""
        
        # Base compose structure
        compose_content = f"""version: '3.8'
services:
  {self.config['service_name']}:
    build:
      context: .
      dockerfile: {self.config['dockerfile']}
    ports:
      - "8000:8000"
    volumes:
      - /Volumes/ssd/models:/models  # Model cache directory
      - ./output:/app/output         # Output directory
    environment:"""
        
        # Add environment variables
        for key, value in self.config['environment'].items():
            compose_content += f"\n      - {key}={value}"
        
        # Add common environment variables
        compose_content += """
      - TRANSFORMERS_CACHE=/models
      - HF_HOME=/models
      - PYTHONUNBUFFERED=1"""
        
        # Add GPU runtime for CUDA
        if self.config['dockerfile'] == 'Dockerfile.cuda':
            compose_content += """
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]"""
        
        # Add memory limits
        if self.config['memory_limit']:
            compose_content += f"""
    mem_limit: {self.config['memory_limit']}
    memswap_limit: {self.config['memory_limit']}"""
        
        # Add restart policy
        compose_content += """
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

networks:
  default:
    driver: bridge"""
        
        return compose_content
    
    def generate_tensordock_compose(self) -> str:
        """Generate TensorDock-optimized docker-compose.yml."""
        
        compose_content = f"""version: '3.8'
services:
  music-generator:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models              # Model cache on persistent storage
      - ./output:/app/output          # Output directory
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - TRANSFORMERS_CACHE=/models
      - HF_HOME=/models
      - PYTHONUNBUFFERED=1
      - OMP_NUM_THREADS=4
      - MKL_NUM_THREADS=4
      - NUMEXPR_NUM_THREADS=4
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    mem_limit: 16g
    memswap_limit: 16g
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

networks:
  default:
    driver: bridge"""
        
        return compose_content
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.project_root / 'output',
            self.project_root / 'models',
            self.project_root / 'scripts'
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def update_requirements(self):
        """Update requirements.txt with GPU-specific packages if needed."""
        requirements_file = self.project_root / 'requirements.txt'
        
        with open(requirements_file, 'r') as f:
            requirements = f.read()
        
        # Add GPU-specific packages for CUDA builds
        if self.config['dockerfile'] == 'Dockerfile.cuda':
            gpu_packages = [
                'nvidia-ml-py3',
                'gputil',
                'accelerate'
            ]
            
            for package in gpu_packages:
                if package not in requirements:
                    requirements += f"\n{package}"
        
        with open(requirements_file, 'w') as f:
            f.write(requirements)
        
        print(f"📝 Updated requirements.txt for {self.config['dockerfile']}")
    
    def save_configuration(self):
        """Save system configuration and deployment info."""
        config_data = {
            'system_info': self.detector.system_info,
            'deployment_config': self.config,
            'deployment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dockerfile_used': self.config['dockerfile'],
            'service_name': self.config['service_name']
        }
        
        config_file = self.project_root / 'deployment_config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Saved deployment configuration to {config_file}")
    
    def build_and_deploy(self, force_rebuild: bool = False):
        """Build and deploy the application."""
        print("🚀 Starting deployment process...")
        
        # Create directories
        self.create_directories()
        
        # Update requirements
        self.update_requirements()
        
        # Generate docker-compose.yml
        compose_content = self.generate_docker_compose()
        compose_file = self.project_root / 'docker-compose.yml'
        
        with open(compose_file, 'w') as f:
            f.write(compose_content)
        
        print(f"📝 Generated {compose_file}")
        
        # Save configuration
        self.save_configuration()
        
        # Build and start services
        try:
            if force_rebuild:
                print("🔨 Forcing rebuild of Docker images...")
                subprocess.run(['docker-compose', 'build', '--no-cache'], 
                             cwd=self.project_root, check=True)
            
            print("🏗️ Building and starting services...")
            subprocess.run(['docker-compose', 'up', '--build', '-d'], 
                         cwd=self.project_root, check=True)
            
            print("✅ Deployment completed successfully!")
            print(f"🌐 API available at: http://localhost:8000")
            print(f"📋 Health check: http://localhost:8000/health")
            print(f"📖 Documentation: http://localhost:8000/docs")
            
            # Wait for service to be ready
            print("⏳ Waiting for service to be ready...")
            time.sleep(10)
            
            # Check health
            try:
                result = subprocess.run(['curl', '-f', 'http://localhost:8000/health'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print("✅ Service is healthy and ready!")
                else:
                    print("⚠️ Service may not be fully ready. Check logs with: docker-compose logs")
            except subprocess.TimeoutExpired:
                print("⚠️ Health check timed out. Service may still be starting.")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Deployment failed: {e}")
            print("🔍 Check Docker logs with: docker-compose logs")
            sys.exit(1)
    
    def deploy_tensordock(self):
        """Deploy specifically optimized for TensorDock."""
        print("🎯 Deploying TensorDock-optimized configuration...")
        
        # Create directories
        self.create_directories()
        
        # Generate TensorDock compose
        compose_content = self.generate_tensordock_compose()
        compose_file = self.project_root / 'docker-compose.tensordock.yml'
        
        with open(compose_file, 'w') as f:
            f.write(compose_content)
        
        print(f"📝 Generated {compose_file}")
        
        # Create TensorDock deployment script
        deploy_script = """#!/bin/bash
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
"""
        
        deploy_script_file = self.project_root / 'deploy_tensordock.sh'
        with open(deploy_script_file, 'w') as f:
            f.write(deploy_script)
        
        deploy_script_file.chmod(0o755)
        
        print(f"📝 Created TensorDock deployment script: {deploy_script_file}")
        print("🚀 Run './deploy_tensordock.sh' on your TensorDock instance")
    
    def print_deployment_summary(self):
        """Print deployment summary."""
        print("\n🎯 Deployment Summary")
        print("=" * 50)
        print(f"Architecture: {self.detector.system_info['architecture']}")
        print(f"Dockerfile: {self.config['dockerfile']}")
        print(f"Service: {self.config['service_name']}")
        print(f"Recommended Model: {self.config['recommended_model']}")
        print(f"Max Workers: {self.config['max_workers']}")
        print(f"Memory Limit: {self.config['memory_limit']}")
        
        if self.detector.system_info['gpu']['available']:
            print(f"GPU: ✅ {self.detector.system_info['gpu']['type'].upper()}")
            print(f"GPU Memory: {self.detector.system_info['gpu']['memory']} MB")
        else:
            print("GPU: ❌ CPU Only")

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Smart Music Generation API Deployer')
    parser.add_argument('--detect-only', action='store_true', 
                       help='Only detect system, don\'t deploy')
    parser.add_argument('--force-rebuild', action='store_true', 
                       help='Force rebuild of Docker images')
    parser.add_argument('--tensordock', action='store_true', 
                       help='Generate TensorDock-optimized deployment')
    parser.add_argument('--save-config', action='store_true', 
                       help='Save system configuration to file')
    
    args = parser.parse_args()
    
    deployer = SmartDeployer()
    
    if args.detect_only:
        deployer.detector.print_system_info()
        return
    
    if args.save_config:
        deployer.detector.save_config()
        return
    
    if args.tensordock:
        deployer.deploy_tensordock()
        return
    
    # Standard deployment
    deployer.print_deployment_summary()
    
    print("\n🚀 Starting smart deployment...")
    deployer.build_and_deploy(force_rebuild=args.force_rebuild)

if __name__ == "__main__":
    main() 