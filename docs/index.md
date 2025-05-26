```{raw} html
<div class="hero-section">
  <div class="hero-background"></div>
  <h1 class="hero-title">AION-1</h1>
  <p class="hero-subtitle">AstronomIcal Omnimodal Network</p>
  <p class="hero-description">The first large-scale multimodal foundation model for astronomy</p>
  <div class="hero-buttons">
    <a href="#quick-start" class="btn-primary">Get Started →</a>
    <a href="https://arxiv.org/abs/2406.00000" class="btn-secondary">Read the Paper</a>
    <a href="https://colab.research.google.com/github/polymathic-ai/aion/blob/main/notebooks/AION_quickstart.ipynb" class="btn-secondary">Run on Colab</a>
  </div>
</div>
```

# Welcome to AION-1

AION-1 (AstronomIcal Omnimodal Network) represents a breakthrough in astronomical machine learning: the first foundation model capable of understanding and processing arbitrary combinations of astronomical observations across 39 different data modalities. Trained on over 200 million astronomical objects, AION-1 unifies imaging, spectroscopy, photometry, and catalog data from major ground- and space-based observatories into a single, powerful framework.

## 🌟 Why AION-1?

Traditional approaches in astronomy treat each data modality in isolation, missing the rich interconnections between different types of observations. AION-1 fundamentally changes this paradigm by:

- **Learning Cross-Modal Relationships**: The model discovers how different observations relate to each other, building a deep understanding of the underlying astrophysical objects
- **Enabling Flexible Data Fusion**: Scientists can use any combination of available observations without redesigning their analysis pipeline
- **Excelling in Low-Data Regimes**: AION-1 achieves competitive results with orders of magnitude less labeled data than supervised approaches
- **Providing Universal Representations**: The learned embeddings capture physically meaningful structure useful across diverse downstream tasks

## 📊 Key Capabilities

```{eval-rst}
.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: 🌌 39 Data Modalities
      :class-card: feature-card

      Seamlessly integrates multiband images, optical spectra, photometry, and catalog data from HSC, Legacy Survey, SDSS, DESI, and Gaia

   .. grid-item-card:: 🧠 200M+ Objects
      :class-card: feature-card

      Pre-trained on massive astronomical datasets spanning galaxies, stars, and quasars across multiple surveys

   .. grid-item-card:: 🔧 Flexible Architecture
      :class-card: feature-card

      Two-stage design with modality-specific tokenization followed by transformer-based multimodal masked modeling

   .. grid-item-card:: ⚡ Emergent Behaviors
      :class-card: feature-card

      Demonstrates physical understanding, superior low-data performance, and meaningful latent space organization

   .. grid-item-card:: 🎯 Versatile Applications
      :class-card: feature-card

      Supports regression, classification, generation, retrieval, and cross-modal prediction tasks out-of-the-box

   .. grid-item-card:: 🌍 Open Science
      :class-card: feature-card

      Fully open-source including datasets, training scripts, and model weights for reproducible research
```

## 🚀 Quick Start

Getting started with AION-1 is straightforward:

```python
# Minimal end-to-end example
from aion import AION
import numpy as np

# 1) Load a pre-trained checkpoint (800 M parameters)
model = AION.from_pretrained('polymathic-ai/aion-base')

# 2) Prepare demo inputs (96×96 HSC g,r,i,z,y cut-out and SDSS spectrum)
galaxy_image = np.load('hsc_cutout_5band.npy')       # shape (5,96,96)
galaxy_spectrum = np.load('sdss_spectrum.npy')       # dict with wavelength/flux

# 3) Generate a high-resolution DESI-like spectrum from the image
generated = model.generate(
    inputs={'image': galaxy_image},
    targets=['spectrum']
)

# 4) Extract joint embeddings for downstream use
embeddings = model.encode({'image': galaxy_image, 'spectrum': galaxy_spectrum})
```

## 🔬 Scientific Impact

AION-1 demonstrates several emergent behaviors that reflect its deep understanding of astronomical data:

### Physical Understanding
- Solves non-trivial scientific tasks using only simple linear probes on learned representations
- Organizes objects in embedding space along physically meaningful dimensions
- Captures relationships between disparate observations of the same physical phenomena

### Performance Advantages
- Achieves state-of-the-art results on galaxy property estimation, stellar parameter prediction, and morphology classification
- Outperforms supervised baselines by 3x on rare object detection tasks
- Enables accurate cross-modal prediction even for modality pairs never seen during training

### Practical Benefits
- Reduces data requirements by orders of magnitude for downstream tasks
- Enables seamless integration of heterogeneous observations
- Provides robust uncertainty quantification through multiple sampling

## 📚 Documentation Overview

```{eval-rst}
.. grid:: 2 2 2 4
   :gutter: 3

   .. grid-item-card:: Installation & Setup
      :link: installation.html
      :class-card: doc-card

      Environment setup, dependencies, and configuration

   .. grid-item-card:: Model Architecture
      :link: architecture.html
      :class-card: doc-card

      Deep dive into tokenization, transformers, and design

   .. grid-item-card:: Usage Guide
      :link: usage.html
      :class-card: doc-card

      Tutorials, examples, and best practices

   .. grid-item-card:: API Reference
      :link: api.html
      :class-card: doc-card

      Complete API documentation and method signatures
```

```{toctree}
:hidden:
:maxdepth: 2

installation
architecture
usage
api
```

## 🤝 Join the Community

```{raw} html
<div class="community-section">
  <h3>Advancing astronomical AI together</h3>
  <p>AION-1 is developed by Polymathic AI in collaboration with the Flatiron Institute and leading astronomical institutions worldwide. We welcome contributions from astronomers, ML researchers, and data scientists interested in pushing the boundaries of multimodal scientific machine learning.</p>
  <a href="contributing.html" class="btn-primary">Start Contributing →</a>
</div>
```
