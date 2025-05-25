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
pip install aion
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
