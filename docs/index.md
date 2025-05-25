```{raw} html
<div class="hero-section">
  <div class="hero-background"></div>
  <h1 class="hero-title">AION-1</h1>
  <p class="hero-subtitle">AstronomIcal Omnimodal Network</p>
  <p class="hero-description">Next-generation foundation model for multimodal astronomical analysis</p>
  <div class="hero-buttons">
    <a href="#quick-start" class="btn-primary">Get Started →</a>
    <a href="https://github.com/polymathic-ai/aion" class="btn-secondary">View on GitHub</a>
  </div>
</div>
```

# Welcome to the AION-1 documentation



## 🚀 Quick Start

```{admonition} Get up and running with AION
:class: tip

Our foundation model seamlessly processes astronomical imaging, spectroscopy, and catalog data.
```

```python
from aion import AION

# Initialize the model
model = AION.from_pretrained('polymathic-ai/aion-base')

# Process multimodal astronomical data
outputs = model.generate(
    images=galaxy_images,
    spectra=stellar_spectra,
    catalog=source_catalog
)
```

## ✨ Key Capabilities

```{eval-rst}
.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: 🌌 Multimodal Processing
      :class-card: feature-card

      Unified handling of images, spectra, time series, and catalog data through specialized encoders

   .. grid-item-card:: 🧠 Foundation Architecture
      :class-card: feature-card

      State-of-the-art transformer backbone pre-trained on massive astronomical datasets

   .. grid-item-card:: 🔧 Extensible Framework
      :class-card: feature-card

      Modular codec system allows easy integration of new data modalities and instruments

   .. grid-item-card:: ⚡ High Performance
      :class-card: feature-card

      Optimized for both research and production with efficient batching and GPU acceleration

   .. grid-item-card:: 📊 Rich Embeddings
      :class-card: feature-card

      Generate powerful representations for downstream tasks like classification and discovery

   .. grid-item-card:: 🌍 Community Driven
      :class-card: feature-card

      Open-source development with contributions from leading astronomical institutions
```

## 📚 Documentation

```{eval-rst}
.. grid:: 2 2 2 4
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation.html
      :class-card: doc-card

      Quick setup guide and requirements

   .. grid-item-card:: Architecture
      :link: architecture.html
      :class-card: doc-card

      Deep dive into model design

   .. grid-item-card:: Usage Guide
      :link: usage.html
      :class-card: doc-card

      Examples and best practices

   .. grid-item-card:: API Reference
      :link: api.html
      :class-card: doc-card

      Complete API documentation
```

```{toctree}
:hidden:
:maxdepth: 2

installation
architecture
usage
api
contributing
```

## 🤝 Join the Community

```{raw} html
<div class="community-section">
  <h3>Advancing astronomical AI together</h3>
  <p>AION is developed by Polymathic AI in collaboration with astronomers and ML researchers worldwide. Join us in building the future of astronomical data analysis.</p>
  <a href="contributing.html" class="btn-primary">Start Contributing →</a>
</div>
```
