# 🌌 AION-1: AstronomIcal Omnimodal Network

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.4.0-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://github.com/PolymathicAI/AION/actions/workflows/test.yaml/badge.svg)](https://github.com/PolymathicAI/AION/actions/workflows/test.yaml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/polymathic-ai/aion-base)

**Polymathic's Large Omnimodal Model for Astronomy**

[🚀 Quick Start](#-quick-start) • [🎓 Tutorials](#-tutorials) • [🔬 Scientific Overview](#-scientific-overview) • [📦 Advanced Installation](#-advanced-installation)

</div>

---

## 🎯 Overview

<div align="center">
    <img src="assets/aion.png" alt="AION Logo" width="600">
</div>

AION-1 is a cutting-edge large omnimodal model specifically designed for astronomical surveys. It seamlessly integrates multiple data modalities, and enables simple adaptation to a wide range of astronomical tasks.

## 🚀 Quick Start

Assuming you have PyTorch installed, you can install AION trivially with:
```bash
pip install polymathic-aion
```

Then you can load the pretrained model and start analyzing astronomical data:
```python
import torch
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

# Load model and codec manager
model = AION.from_pretrained('aion-base').to('cuda')  # or 'aion-large', 'aion-xlarge'
codec_manager = CodecManager(device='cuda')

# Prepare your astronomical data (example: Legacy Survey image)
image = LegacySurveyImage(
    flux=your_image_tensor,  # Shape: [batch, 4, height, width] for g,r,i,z bands
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)

# Encode data to tokens
tokens = codec_manager.encode(image)

# Option 1: Extract embeddings for downstream tasks
embeddings = model.encode(tokens, num_encoder_tokens=600)

# Option 2: Generate predictions (e.g., redshift)
from aion.modalities import Z
preds = model(
    codec_manager.encode(image),
    target_modality=Z,
)
```

## 🎓 Tutorials

Start with our interactive tutorial:
- **[Open in Google Colab](https://colab.research.google.com/github/PolymathicAI/AION/blob/main/notebooks/Tutorial.ipynb)** - Learn AION basics interactively, no local setup required!


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

| **Variant** | **Encoder Blocks** | **Decoder Blocks** | **Model Dim** | **Heads** | **Total Params** | **Model** |
|------------:|-------------------:|-------------------:|--------------:|----------:|-----------------:|-----------|
| **Base**    | 12                 | 12                 | 768           | 12        | 300 M            | [aion-base](https://huggingface.co/polymathic-ai/aion-base) |
| **Large**   | 24                 | 24                 | 1024          | 16        | 800 M            | soon |
| **XLarge**  | 24                 | 24                 | 2048          | 32        | 3 B              | soon |

> **Pretraining**
> – Global batch size: 8 192
> – Steps: Base (1.5 days on 64 H100), Large (2.5 days on 100 H100), XLarge (3.5 days on 288 H100)
> – Optimizer: AdamW, peak LR 2 × 10⁻⁴, linear warmup + cosine decay

## 🔧 Data Preparation

AION uses a typed data system to understand the provenance of each astronomical observation. Each modality must be properly formatted:

### Modality Types
```python
from aion.modalities import (
    LegacySurveyImage, HSCImage,           # Images
    DESISpectrum, SDSSSpectrum,            # Spectra
    LegacySurveyFluxG, HSCMagG,            # Photometry
    GaiaParallax, Z,                       # Scalars
    # ... and 30+ more modalities
)
```

### Example: Preparing Legacy Survey Data
```python
import torch
from aion.modalities import LegacySurveyImage, LegacySurveyFluxG

# Format image data (shape: [batch, 4, height, width])
image = LegacySurveyImage(
    flux=torch.tensor(image_data, dtype=torch.float32),
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)

# Format scalar photometry
flux_g = LegacySurveyFluxG(value=torch.tensor([flux_values]))
```

### Supported Data Formats
| Survey | Modality | Required Format |
|--------|----------|-----------------|
| **Legacy Survey** | Images | 4-band (g,r,i,z), any resolution (auto-cropped to 96×96) |
| **HSC** | Images | 5-band (g,r,i,z,y), any resolution |
| **DESI/SDSS** | Spectra | Flux, inverse variance, wavelength arrays |
| **Gaia** | BP/RP | Coefficient arrays (55 coefficients each) |
| **All Surveys** | Scalars | Single values or 1D tensors |

---

## 💡 Example Use Cases

### 🔍 Similarity Search
Find galaxies similar to a query object across different modalities:
```python
# Extract embeddings for similarity search
query_embedding = model.encode(codec_manager.encode(query_image))
all_embeddings = model.encode(codec_manager.encode(*dataset_images))

# Find most similar objects using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(query_embedding, all_embeddings)
similar_objects = similarity_scores.argsort()[::-1][:10]  # Top 10 similar
```

### 📊 Property Prediction
Build lightweight models on AION embeddings:
```python
# Extract embeddings from multiple modalities
embeddings = model.encode(codec_manager.encode(
    image, spectrum, flux_g, flux_r, flux_i, flux_z
), num_encoder_tokens=900)

# Train simple regressor for stellar mass, redshift, etc.
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(embeddings.mean(axis=1), target_property)
```

### 🌌 Generative Modeling
Predict missing astronomical properties:
```python
# Predict redshift from photometry + morphology
predictions = model(
    codec_manager.encode(image, flux_g, flux_r, flux_i, flux_z),
    target_mask={'tok_z': torch.zeros(batch_size, 1)},
    num_encoder_tokens=600
)
redshift_probs = torch.softmax(predictions['tok_z'], dim=-1)
```

## 📦 Advanced Installation

AION offers flexible installation options to suit your environment and requirements.

To install AION with PyTorch included:

```bash
pip install polymathic-aion[torch]
```

For contributors and developers:

```bash
pip install polymathic-aion[torch,dev]
```

This includes testing frameworks, linting tools, and development dependencies.

For specific PyTorch versions (e.g., CUDA support):

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Then install AION
pip install polymathic-aion
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

AION is developed by [Polymathic AI](https://polymathic-ai.org/), advancing the frontier of AI for scientific applications. We would like to acknowledge the support of the Simons Foundation and of Schmidt Sciences. This project was provided with computer and storage resources by GENCI at IDRIS thanks to the grant 2024-GC011015468 on the supercomputer
Jean Zay’s H100 partition. Additionally, some of the computations in this work were run at facilities supported by the Scientific Computing Core at the Flatiron Institute, a division of the Simons Foundation.

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/PolymathicAI/AION/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PolymathicAI/AION/discussions)

---

<div align="center">
  <sub>Built with ❤️ for the astronomical community</sub>
</div>
