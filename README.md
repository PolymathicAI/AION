# 🌌 AION-1: AstronomIcal Omnimodal Network

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.4.0-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://github.com/PolymathicAI/AION/actions/workflows/test.yaml/badge.svg)](https://github.com/PolymathicAI/AION/actions/workflows/test.yaml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb)

**Polymathic's Large Omnimodal Model for Astronomy**

[🚀 Quick Start](#-quick-start) • [🔬 Scientific Overview](#-scientific-overview) • [📚 Documentation](#-documentation) • [📦 Advanced Installation](#-advanced-installation) • [🤝 Contributing](#-contributing)

</div>

---

## 🎯 Overview

AION-1 is a cutting-edge large omnimodal model specifically designed for astronomical surveys. It seamlessly integrates multiple data modalities, and enables simple adaptation to a wide range of astronomical tasks.


## 🚀 Quick Start

Assuming you have PyTorch installed, you can install AION trivially with:
```bash
pip install aion
```

Then you can load the pretrained model and start using it:
```python
from aion import AION

# Load the pretrained model
model = AION.from_pretrained('aion-base')

# Your astronomical analysis begins here!
```

## 🔬 Scientific Overview

### 🧬 Architecture
AION-1 employs a two-stage, transformer-based design:
1. **Modality-Specific Tokenizers** transform raw inputs into discrete tokens
2. **Unified Encoder–Decoder Transformer** ingests all token streams via a multimodal masked modeling (4M) objective

---

### 🗂️ Supported Modalities
AION-1’s tokenizers cover **39 distinct data types**, grouped by survey and data category

| **Category**            | **Description**                         | **Token Name(s)**        |
|-------------------------|-----------------------------------------|--------------------------|
| **Imaging (2)**         | Legacy Survey, HSC Wide                 | `tok_image_ls`, `tok_image_hsc` |
| **Catalog (1)**         | Legacy Survey catalog entries           | `catalog`                |
| **Spectra (2)**         | SDSS, DESI                              | `tok_spectrum_sdss`, `tok_spectrum_desi` |
| **Gaia (4)**            | BP/RP spectra, parallax, sky coords     | `tok_xp_bp`, `tok_xp_rp`, `tok_parallax`, `tok_ra`, `tok_dec` |
| **Gaia Photometry (3)** | G/BP/RP flux                            | `tok_flux_g_gaia`, `tok_flux_bp_gaia`, `tok_flux_rp_gaia` |
| **Legacy Survey (9)**   | g,r,i,z bands & WISE W1–W4 flux, E(B–V) | `tok_flux_g`,…,`tok_flux_w4`, `tok_ebv` |
| **Legacy Shape (3)**    | Ellipticity components & effective radius | `tok_shape_e1`, `tok_shape_e2`, `tok_shape_r` |
| **HSC Photometry (5)**  | g,r,i,z,y magnitudes                    | `tok_mag_g`,…,`tok_mag_y` |
| **HSC Extinction (5)**  | g,r,i,z,y extinctions                   | `tok_a_g`,…,`tok_a_y`    |
| **HSC Shape (3)**       | Shape components 11,22,12               | `tok_shape11`, `tok_shape22`, `tok_shape12` |
| **Other (1)**           | Spectroscopic redshift                  | `tok_z`                  |

---

### 📈 Model Variants

| **Variant** | **Encoder Blocks** | **Decoder Blocks** | **Model Dim** | **Heads** | **Total Params** |
|------------:|-------------------:|-------------------:|--------------:|----------:|-----------------:|
| **Base**    | 12                 | 12                 | 768           | 12        | 300 M            |
| **Large**   | 24                 | 24                 | 1024          | 16        | 800 M            |
| **XLarge**  | 24                 | 24                 | 2048          | 32        | 3 B              |

> **Pretraining**
> – Global batch size: 8 192
> – Steps: Base (1.5 days on 64 H100), Large (2.5 days on 100 H100), XLarge (3.5 days on 288 H100)
> – Optimizer: AdamW, peak LR 2 × 10⁻⁴, linear warmup + cosine decay

## 📚 Documentation

### 🎓 Tutorials

Start with our interactive tutorial:
- **[Open in Google Colab](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb)** - Learn AION basics interactively, no local setup required!

For detailed guides, see the [online documentation](https://polymathic-ai.github.io/AION/).

## 📦 Advanced Installation

AION offers flexible installation options to suit your environment and requirements.

To install AION with PyTorch included:

```bash
pip install aion[torch]
```

For contributors and developers:

```bash
pip install aion[torch,dev]
```

This includes testing frameworks, linting tools, and development dependencies.

For specific PyTorch versions (e.g., CUDA support):

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Then install AION
pip install aion
```

## 🤝 Contributing

We welcome contributions from the astronomical and ML communities!

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies:
   ```bash
   pip install -e .[torch,dev]
   ```
4. Make your changes and ensure tests pass:
   ```bash
   pytest
   ```
5. Run linting:
   ```bash
   ruff check .
   ```
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

AION is developed by [Polymathic AI](https://polymathic-ai.org/), advancing the frontier of AI for scientific discovery.

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/PolymathicAI/AION/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PolymathicAI/AION/discussions)

---

<div align="center">
  <sub>Built with ❤️ for the astronomical community</sub>
</div>
