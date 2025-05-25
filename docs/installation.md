# Installation Guide

This comprehensive guide will walk you through installing AION-1 and setting up your environment for astronomical multimodal analysis.

## System Requirements

### Hardware Requirements

AION-1 is designed to run efficiently on various hardware configurations:

- **Minimum Requirements**:
  - CPU: 4+ cores (Intel/AMD x86_64 or Apple Silicon)
  - RAM: 16 GB
  - GPU: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
  - Storage: 50 GB free space for models and data

- **Recommended Requirements**:
  - CPU: 8+ cores
  - RAM: 32 GB or more
  - GPU: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, A5000, or better)
  - Storage: 100 GB+ free space

- **For Large-Scale Processing**:
  - Multiple GPUs with NVLink
  - 64GB+ RAM
  - Fast SSD storage for data loading

### Software Requirements

- Python 3.10 or later
- CUDA 11.8+ (for GPU support)
- Operating System: Linux, macOS, or Windows

## Installation Methods

### 1. Quick Install via PyPI

The simplest way to install AION-1 is through PyPI:

```bash
pip install aion
```

This installs the core AION package with minimal dependencies.

### 2. Full Installation with PyTorch

For GPU support and optimal performance:

```bash
# Install PyTorch first (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install AION
pip install aion[full]
```

### 3. Development Installation

For contributors or those who want the latest features:

```bash
# Clone the repository
git clone https://github.com/polymathic-ai/aion.git
cd aion

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 4. Docker Installation

For containerized deployments:

```bash
# Pull the official Docker image
docker pull polymathic/aion:latest

# Run with GPU support
docker run --gpus all -it polymathic/aion:latest
```

## Setting Up Your Environment

### 1. Virtual Environment Setup

We strongly recommend using a virtual environment:

```bash
# Using venv
python -m venv aion-env
source aion-env/bin/activate  # On Windows: aion-env\Scripts\activate

# Using conda
conda create -n aion python=3.10
conda activate aion
```

### 2. Verify Installation

After installation, verify everything is working:

```python
import aion
import torch

# Check AION version
print(f"AION version: {aion.__version__}")

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test loading a model
from aion import AION
model = AION.from_pretrained('polymathic-ai/aion-tiny')
print("Model loaded successfully!")
```

### 3. Download Pre-trained Models

AION-1 comes in three sizes. Models are automatically downloaded on first use, but you can pre-download them:

```python
from aion import AION

# Download models (choose based on your hardware)
model_tiny = AION.from_pretrained('polymathic-ai/aion-tiny')    # 300M parameters
model_base = AION.from_pretrained('polymathic-ai/aion-base')    # 800M parameters
model_large = AION.from_pretrained('polymathic-ai/aion-large')  # 3.1B parameters
```

Model sizes and requirements:
- **aion-tiny**: ~1.2 GB, runs on 8GB GPUs
- **aion-base**: ~3.2 GB, recommended 16GB+ GPU
- **aion-large**: ~12 GB, requires 24GB+ GPU

### 4. Configure Model Cache

By default, models are cached in `~/.cache/huggingface/hub/`. To change this:

```bash
# Set environment variable
export HF_HOME=/path/to/your/cache

# Or in Python
import os
os.environ['HF_HOME'] = '/path/to/your/cache'
```

## Installing Optional Dependencies

### For Astronomical Data Processing

```bash
pip install astropy fits
```

### For Visualization

```bash
pip install matplotlib seaborn plotly
```

### For Advanced Scientific Computing

```bash
pip install scipy scikit-learn pandas
```

## Platform-Specific Instructions

### Linux

Most straightforward installation. Ensure you have:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# CentOS/RHEL
sudo yum install python3-devel python3-pip
```

### macOS

For Apple Silicon Macs:
```bash
# Install using conda for better compatibility
conda install pytorch torchvision -c pytorch
pip install aion
```

Note: GPU acceleration on macOS uses Metal Performance Shaders (MPS):
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Windows

Ensure you have Visual C++ Build Tools:
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install with "Desktop development with C++"

Then:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install aion
```

## Troubleshooting

### Common Issues and Solutions

**1. CUDA Out of Memory**
```python
# Reduce batch size
model.eval()
with torch.no_grad():
    outputs = model(inputs)

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

**2. Import Errors**
```bash
# Ensure all dependencies are installed
pip install --upgrade aion[full]

# Check for conflicts
pip check
```

**3. Slow Model Loading**
```python
# Use faster model loading
model = AION.from_pretrained('polymathic-ai/aion-base',
                            torch_dtype=torch.float16,
                            device_map="auto")
```

**4. Version Conflicts**
```bash
# Create a fresh environment
conda create -n aion-clean python=3.10
conda activate aion-clean
pip install aion
```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/polymathic-ai/aion/issues)
2. Join our [Discord community](https://discord.gg/polymathic-ai)
3. Consult the [FAQ section](https://polymathic-ai.org/aion/faq)

## Next Steps

Now that you have AION-1 installed, explore:
- [Architecture Overview](architecture.html) - Understand how AION-1 works
- [Usage Guide](usage.html) - Learn to use AION-1 for your research
- [API Reference](api.html) - Detailed API documentation

## Performance Optimization

### GPU Memory Management

```python
# Clear cache when switching between models
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Optimize for inference
model.eval()
torch.set_grad_enabled(False)
```

### Multi-GPU Setup

```python
# DataParallel for simple multi-GPU
model = torch.nn.DataParallel(model)

# DistributedDataParallel for better performance
import torch.distributed as dist
dist.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
```

### CPU Optimization

```python
# Enable MKL optimizations
torch.set_num_threads(8)  # Adjust based on your CPU

# Use channels_last memory format
model = model.to(memory_format=torch.channels_last)
```
