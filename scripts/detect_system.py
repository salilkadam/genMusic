#!/usr/bin/env python3
"""
System Detection Script for Music Generation API
Detects system architecture and GPU capabilities to choose optimal Docker configuration.
"""

import platform
import subprocess
import sys
import os
import json
from typing import Dict, Any, Optional

class SystemDetector:
    def __init__(self):
        self.system_info = {}
        self.detect_all()
    
    def detect_architecture(self) -> str:
        """Detect CPU architecture."""
        arch = platform.machine().lower()
        
        if arch in ['arm64', 'aarch64']:
            return 'arm64'
        elif arch in ['x86_64', 'amd64']:
            return 'x86_64'
        elif arch in ['i386', 'i686']:
            return 'x86'
        else:
            return arch
    
    def detect_os(self) -> str:
        """Detect operating system."""
        return platform.system().lower()
    
    def detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU availability and type."""
        gpu_info = {
            'available': False,
            'type': None,
            'count': 0,
            'memory': 0,
            'cuda_available': False,
            'cuda_version': None
        }
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=count,memory.total', '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info['available'] = True
                gpu_info['type'] = 'nvidia'
                gpu_info['count'] = len(lines)
                gpu_info['memory'] = sum(int(line.split(',')[1]) for line in lines)
                
                # Check CUDA version
                cuda_result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                           capture_output=True, text=True, timeout=10)
                if cuda_result.returncode == 0:
                    gpu_info['cuda_available'] = True
                    gpu_info['cuda_version'] = cuda_result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check for AMD GPU (ROCm)
        try:
            result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and not gpu_info['available']:
                gpu_info['available'] = True
                gpu_info['type'] = 'amd'
                # Could add more AMD-specific detection here
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return gpu_info
    
    def detect_docker(self) -> Dict[str, Any]:
        """Detect Docker availability and configuration."""
        docker_info = {
            'available': False,
            'version': None,
            'compose_available': False,
            'buildx_available': False
        }
        
        try:
            # Check Docker
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                docker_info['available'] = True
                docker_info['version'] = result.stdout.strip()
            
            # Check Docker Compose
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                docker_info['compose_available'] = True
            
            # Check Docker Buildx
            result = subprocess.run(['docker', 'buildx', 'version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                docker_info['buildx_available'] = True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return docker_info
    
    def detect_memory(self) -> Dict[str, Any]:
        """Detect system memory."""
        memory_info = {
            'total_gb': 0,
            'available_gb': 0
        }
        
        try:
            if self.system_info.get('os') == 'linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            memory_info['total_gb'] = int(line.split()[1]) / (1024 * 1024)
                        elif line.startswith('MemAvailable:'):
                            memory_info['available_gb'] = int(line.split()[1]) / (1024 * 1024)
            elif self.system_info.get('os') == 'darwin':
                result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                if result.returncode == 0:
                    memory_info['total_gb'] = int(result.stdout.split()[1]) / (1024 ** 3)
        except Exception:
            pass
        
        return memory_info
    
    def detect_cpu_cores(self) -> int:
        """Detect number of CPU cores."""
        try:
            return os.cpu_count() or 1
        except:
            return 1
    
    def detect_all(self):
        """Detect all system information."""
        self.system_info = {
            'architecture': self.detect_architecture(),
            'os': self.detect_os(),
            'gpu': self.detect_gpu(),
            'docker': self.detect_docker(),
            'memory': self.detect_memory(),
            'cpu_cores': self.detect_cpu_cores(),
            'hostname': platform.node(),
            'python_version': platform.python_version()
        }
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Determine optimal Docker configuration based on system."""
        config = {
            'dockerfile': None,
            'service_name': None,
            'recommended_model': 'small',
            'max_workers': 1,
            'timeout_multiplier': 1.0,
            'memory_limit': None,
            'environment': {}
        }
        
        arch = self.system_info['architecture']
        gpu = self.system_info['gpu']
        memory = self.system_info['memory']
        
        # Choose Dockerfile based on capabilities
        if gpu['available'] and gpu['type'] == 'nvidia' and gpu['cuda_available']:
            config['dockerfile'] = 'Dockerfile.cuda'
            config['service_name'] = 'music-generator-cuda'
            config['recommended_model'] = 'large' if gpu['memory'] > 8000 else 'medium'
            config['max_workers'] = 2
            config['timeout_multiplier'] = 0.3  # GPU is much faster
            config['environment']['CUDA_VISIBLE_DEVICES'] = '0'
        elif arch == 'arm64':
            config['dockerfile'] = 'Dockerfile.apple-silicon'
            config['service_name'] = 'music-generator-apple-silicon'
            config['recommended_model'] = 'small'
            config['max_workers'] = 1
        elif arch == 'x86_64':
            config['dockerfile'] = 'Dockerfile.intel'
            config['service_name'] = 'music-generator-intel'
            config['recommended_model'] = 'small'
            config['max_workers'] = min(2, self.system_info['cpu_cores'] // 2)
        else:
            config['dockerfile'] = 'Dockerfile.intel'  # Default fallback
            config['service_name'] = 'music-generator-default'
        
        # Adjust based on memory
        if memory['total_gb'] > 16:
            config['memory_limit'] = '8g'
        elif memory['total_gb'] > 8:
            config['memory_limit'] = '4g'
        else:
            config['memory_limit'] = '2g'
            config['recommended_model'] = 'small'
        
        # Add optimization environment variables
        config['environment'].update({
            'OMP_NUM_THREADS': str(min(4, self.system_info['cpu_cores'])),
            'MKL_NUM_THREADS': str(min(4, self.system_info['cpu_cores'])),
            'NUMEXPR_NUM_THREADS': str(min(4, self.system_info['cpu_cores']))
        })
        
        return config
    
    def print_system_info(self):
        """Print detailed system information."""
        print("🔍 System Detection Results")
        print("=" * 50)
        print(f"Architecture: {self.system_info['architecture']}")
        print(f"OS: {self.system_info['os']}")
        print(f"CPU Cores: {self.system_info['cpu_cores']}")
        print(f"Memory: {self.system_info['memory']['total_gb']:.1f} GB")
        print(f"Hostname: {self.system_info['hostname']}")
        print(f"Python: {self.system_info['python_version']}")
        
        print("\n🎮 GPU Information")
        print("-" * 30)
        gpu = self.system_info['gpu']
        if gpu['available']:
            print(f"GPU Available: ✅ {gpu['type'].upper()}")
            print(f"GPU Count: {gpu['count']}")
            print(f"GPU Memory: {gpu['memory']} MB")
            if gpu['cuda_available']:
                print(f"CUDA Available: ✅ {gpu['cuda_version']}")
        else:
            print("GPU Available: ❌ CPU Only")
        
        print("\n🐳 Docker Information")
        print("-" * 30)
        docker = self.system_info['docker']
        if docker['available']:
            print(f"Docker: ✅ {docker['version']}")
            print(f"Docker Compose: {'✅' if docker['compose_available'] else '❌'}")
            print(f"Docker Buildx: {'✅' if docker['buildx_available'] else '❌'}")
        else:
            print("Docker: ❌ Not Available")
        
        print("\n🎯 Recommended Configuration")
        print("-" * 30)
        config = self.get_optimal_config()
        print(f"Dockerfile: {config['dockerfile']}")
        print(f"Service: {config['service_name']}")
        print(f"Recommended Model: {config['recommended_model']}")
        print(f"Max Workers: {config['max_workers']}")
        print(f"Memory Limit: {config['memory_limit']}")
        print(f"Timeout Multiplier: {config['timeout_multiplier']:.1f}x")
    
    def save_config(self, filename: str = 'system_config.json'):
        """Save detected configuration to file."""
        config = {
            'system_info': self.system_info,
            'optimal_config': self.get_optimal_config(),
            'detected_at': platform.platform()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Configuration saved to {filename}")

def main():
    """Main function for command-line usage."""
    detector = SystemDetector()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--json':
            print(json.dumps(detector.system_info, indent=2))
        elif sys.argv[1] == '--config':
            print(json.dumps(detector.get_optimal_config(), indent=2))
        elif sys.argv[1] == '--save':
            detector.save_config()
        else:
            detector.print_system_info()
    else:
        detector.print_system_info()

if __name__ == "__main__":
    main() 