# AION-1 Architecture

This document provides a comprehensive overview of AION-1's architecture, explaining how it achieves unified multimodal understanding of astronomical data through innovative tokenization strategies and transformer-based learning.

## Overview

AION-1 employs a two-stage architecture that elegantly handles the complexity of astronomical data:

1. **Universal Tokenization**: Modality-specific encoders convert heterogeneous astronomical observations into discrete tokens
2. **Multimodal Masked Modeling**: A unified transformer learns cross-modal relationships through masked token prediction

This design enables AION-1 to process 39 different data modalities from 5 major astronomical surveys, learning from over 200 million objects.

## Core Design Principles

### 1. Purely Observational Learning

Unlike many scientific ML models, AION-1 is trained exclusively on raw observational data without any labels derived from simulations or physical models. This approach provides:

- **Model-agnostic representations**: Not tied to specific physical assumptions
- **Flexibility**: Can adapt to changing theoretical models
- **Robustness**: Learns patterns directly from data

### 2. Arbitrary Modality Combinations

AION-1 can process any subset of its 39 supported modalities without architectural changes:

- No fixed input requirements
- Graceful handling of missing data
- Dynamic modality fusion

### 3. Scalable Token-Based Approach

By converting all data to tokens, AION-1 achieves:

- Uniform processing across modalities
- Efficient batching and computation
- Natural handling of variable-length inputs

## Stage 1: Universal Tokenization

The tokenization stage addresses a fundamental challenge: how to convert diverse astronomical measurements (images, spectra, scalars) into a common representation suitable for transformer processing.

### Image Tokenization

AION-1's image tokenizer handles multi-band astronomical images from different surveys with varying:
- Resolution and pixel scales
- Number of channels (4-9 bands)
- Noise characteristics
- Dynamic range

#### Architecture
```
# Image tokenizer structure
class ImageCodec:
    - Preprocessing:
        - Center crop to 96x96 pixels
        - Survey-specific rescaling
        - Range compression: arcsinh(flux/α) × β

    - Multi-survey projection:
        - SubsampledLinear layer (9 → 54 channels)
        - Handles variable input bands
        - Embeds survey provenance

    - Encoder: MagVit-based architecture
        - ResNet backbone with 2 compressions
        - Hidden dimensions: 512
        - Bottleneck: 5 dimensions

    - Quantization: Finite Scalar Quantization (FSQ)
        - Levels: [8, 5, 5, 5, 5]
        - Codebook size: 10,000
```

#### Key Innovations

1. **Channel Embedding Scheme**: Accommodates images from different surveys with varying band counts in a single model

2. **Inverse-Variance Weighted Loss**: Leverages known noise properties for optimal reconstruction
   ```
   L_NLL = Σ_i 1/2 || Σ_i^(-1/2) (x_i - Decoder(Encoder(x_i))) ||²
   ```

3. **Survey-Aware Processing**: Maintains provenance information through dedicated embeddings

### Spectrum Tokenization

Astronomical spectra present unique challenges:
- Wavelength ranges vary by instrument (3500-10400 Å)
- Resolution differences (R = 1500-5500)
- Orders of magnitude variation in amplitude

#### Architecture
```
# Spectrum tokenizer structure
class SpectrumCodec:
    - Preprocessing:
        - Median normalization
        - Log-transform median
        - Resampling to latent wavelength grid

    - Latent grid:
        - Range: 3500-10462.4 Å
        - Resolution: 0.8 Å/pixel
        - 8704 pixels total

    - Encoder: ConvNeXt V2
        - Depths: [3, 3, 9, 3]
        - Dimensions: [96, 192, 384, 768]

    - Quantization: Lookup-Free Quantization (LFQ)
        - Embedding dimension: 10
        - Codebook size: 1024
```

#### Spectral Grid Interpolation

The tokenizer uses a shared latent wavelength grid, enabling joint processing of spectra from different instruments:

```python
def to_latent(spectrum, observed_wavelength):
    # Interpolate observed spectrum to latent grid
    return interp1d(observed_wavelength, spectrum, latent_wavelength)
```

### Scalar Tokenization

Scalar quantities (fluxes, shapes, physical parameters) are tokenized using adaptive quantization based on cumulative distribution functions (CDFs).

#### Types of Scalar Quantizers

1. **Linear Quantizer**: For uniformly distributed values
2. **Log Quantizer**: For values spanning orders of magnitude
3. **Reservoir Quantizer**: Learns optimal binning from data
4. **Compressed Quantizer**: Applies transformations before quantization

Example scalar modalities:
- Photometric fluxes (g, r, i, z bands)
- Shape parameters (ellipticity, radius)
- Physical properties (redshift, extinction)

### Token Summary at a Glance

| Modality                                       | Native input tensor shape | Tokens per object | Quantizer type & levels | Codebook size |
|------------------------------------------------|---------------------------|--------------------|-------------------------|---------------|
| Image (HSC / Legacy Survey, 96 × 96 cut-out)   | `(B, N_band, 96, 96)`     | 144 *(18×18 grid)* | FSQ `[8,5,5,5,5]`       | 10 000        |
| Spectrum (SDSS / DESI)                         | `(B, 2, λ)` *(flux,ivar)* | 64 + 1 norm token  | LFQ `dim=10`            | 1 024         |
| Scalar quantity (photometry, shapes, etc.)     | `(B,)`                    | 1 per quantity     | Reservoir (linear/log)  | 256 (default) |
| Catalog (bounding ellipses)                    | `(B, N_obj, 5)`           | ≤100×5             | Composite (per-field)   | mixed         |

These numbers correspond to the default configuration used during pre-training (input budget = 256, output budget = 128 tokens).  They can be modified at fine-tune time as long as the total token budget is respected.

### Catalog Tokenization

Astronomical catalogs contain lists of objects with varying counts per image. AION-1 linearizes these into sequences:

```
# Catalog entry: (X, Y, e1, e2, radius)
# Linearization: Sort by distance from center
# Tokenization: Quantize each component separately
```

## Stage 2: Multimodal Masked Modeling

The second stage uses a transformer encoder-decoder architecture to learn relationships between tokens from different modalities.

### Architecture Details

```
class AION(FourM):
    # Encoder
    - Depth: 12-24 layers (model-dependent)
    - Hidden dimension: 768-2048
    - Attention heads: 12-32
    - MLP ratio: 4.0
    - Activation: SwiGLU

    # Decoder
    - Same architecture as encoder
    - Cross-attention to encoder outputs
    - Modality-specific output heads
```

### Multimodal Masking Strategy

AION-1 uses a sophisticated masking strategy that enables learning both within and across modalities:

1. **Input Token Budget**: Randomly select B tokens across all modalities for input
2. **Output Token Budget**: From remaining tokens, select targets using Beta distribution
3. **Cross-Modal Learning**: Masks ensure model learns to predict any modality from any other

```python
def mask_multimodal(tokens, num_input=256, num_output=128):
    # 1. Select primary modality
    primary_mod = random.choice(modalities)

    # 2. Fill input budget
    input_tokens = sample_tokens(primary_mod, budget=num_input)
    input_tokens += sample_from_other_modalities(remaining_budget)

    # 3. Select outputs (Beta distribution favors fewer tokens)
    num_outputs = sample_beta(alpha=0.1, beta=1.0) * num_output
    output_tokens = sample_from_remaining(num_outputs)

    return input_tokens, output_tokens
```

### Training Objective

The model optimizes a cross-entropy loss over predicted tokens:

```
L = -Σ_t log p(x_t^target | x^observed)
```

This simple objective, combined with diverse masking patterns, enables AION-1 to learn rich cross-modal representations.

## Model Variants

AION-1 comes in three sizes, each using the same architecture with different dimensions:

| Model | Parameters | Encoder Layers | Decoder Layers | Hidden Dim | Attention Heads |
|-------|------------|----------------|----------------|------------|-----------------|
| AION-1-B (Base) | 300M | 12 | 12 | 768 | 12 |
| AION-1-L (Large) | 800M | 24 | 24 | 1024 | 16 |
| AION-1-XL (XLarge) | 3.1B | 24 | 24 | 2048 | 32 |

All models use:
- SwiGLU activation functions
- No bias terms (except in embeddings)
- QK-Norm for training stability
- Rotary position embeddings

## Data Flow Through AION-1

Here's how data flows through the complete pipeline:

```{mermaid}
graph TD
    A[Raw Astronomical Data] --> B[Modality-Specific Preprocessing]
    B --> C[Tokenization]
    C --> D[Token Embeddings + Position Encoding]
    D --> E[Transformer Encoder]
    E --> F[Cross-Modal Representations]
    F --> G[Transformer Decoder]
    G --> H[Modality-Specific Heads]
    H --> I[Predictions/Generations]
```

### Example: Processing Galaxy Data

```python
# 1. Input data
galaxy_data = {
    'image': HSC_5band_image,        # (5, 96, 96)
    'spectrum': SDSS_spectrum,        # (3800,)
    'photometry': flux_measurements   # (8,)
}

# 2. Tokenization
tokens = {
    'image': image_codec.encode(galaxy_data['image']),      # → 144 tokens
    'spectrum': spectrum_codec.encode(galaxy_data['spectrum']), # → 64 tokens
    'photometry': scalar_codec.encode(galaxy_data['photometry']) # → 8 tokens
}

# 3. Embedding and encoding
embeddings = model.embed_inputs(tokens)
encoder_output = model.encode(embeddings)

# 4. Cross-modal generation/prediction
predictions = model.decode(encoder_output, target_modalities)
```

## Key Architectural Innovations

### 1. Modality Embeddings with Provenance

Each token receives two embeddings:
- **Token embedding**: Encodes the discrete token value
- **Modality embedding**: Identifies data type AND source survey

This allows AION-1 to understand that HSC g-band and SDSS g-band images have different characteristics.

### 2. Flexible Attention Patterns

The attention mechanism adapts based on input:
- **Encoder**: Full bidirectional attention across all tokens
- **Decoder**: Causal attention within modalities, cross-attention to encoder

### 3. Hierarchical Token Organization

Tokens are organized hierarchically:
- **Spatial tokens**: Preserve 2D structure for images
- **Sequential tokens**: Maintain order for spectra and catalogs
- **Unordered tokens**: For scalar sets

## Training Infrastructure

### Dataset Construction

AION-1's training leverages pairwise associations between surveys:
- HSC images ↔ SDSS spectra
- SDSS spectra ↔ DESI spectra
- Legacy images ↔ Photometry

This creates a connected graph enabling transitive learning (e.g., HSC → SDSS → DESI).

### Optimization Details

- **Optimizer**: AdamW (β₁=0.9, β₂=0.95, weight decay=0.05)
- **Learning rate**: 2e-4 with cosine decay
- **Warmup**: Linear over first 10% of training
- **Batch size**: 8096 (distributed across GPUs)
- **Training steps**: 205,000
- **Mixed precision**: bfloat16

### Computational Requirements

Training AION-1 requires substantial computational resources:
- **AION-1-B**: 64 H100 GPUs for 1.5 days
- **AION-1-L**: 100 H100 GPUs for 2.5 days
- **AION-1-XL**: 288 H100 GPUs for 3.5 days

## Emergent Capabilities

The architecture enables several emergent behaviors:

### 1. Zero-Shot Cross-Modal Generation
Despite never seeing direct HSC↔DESI associations during training, AION-1 can generate DESI spectra from HSC images through transitive learning.

### 2. Flexible Conditioning
Any modality subset can condition generation of any other subset, enabling:
- Super-resolution (low-res → high-res spectra)
- Cross-modal translation (images → spectra)
- Imputation (partial → complete observations)

### 3. Physically Meaningful Representations
The learned embeddings organize objects along interpretable axes:
- Galaxy types (spiral, elliptical, merger)
- Stellar properties (temperature, metallicity)
- Redshift progression

## Implementation Details

### Memory Efficiency

- **Gradient checkpointing**: Trades computation for memory
- **Mixed precision**: bfloat16 for most operations
- **Efficient attention**: Flash Attention 2 implementation

### Inference Optimization

- **Token caching**: Reuse encoder outputs for multiple decodings
- **Batch processing**: Process multiple objects simultaneously
- **Quantization**: INT8 inference for deployment

## Data Provenance & Licensing

The pre‐training corpus – dubbed *The Multimodal Universe (MMU)* – merges publicly available data products under their respective licences:

| Survey | Release | Reference | Modalities Used |
|--------|---------|-----------|-----------------|
| Legacy Imaging Survey (DECaLS/BASS/MzLS) | DR10 | Dey et al. 2019 | 4-band images, photometry, catalog scalars |
| Hyper Suprime-Cam (HSC) | PDR3 (Wide+Deep) | Aihara et al. 2019 | 5-band images, photometry, shapes |
| Sloan Digital Sky Survey (SDSS) | DR17 | Eisenstein et al. 2011 | R≈2000 spectra |
| Dark Energy Spectroscopic Instrument (DESI) | EDR | DESI Collab. 2023 | R≈3000 spectra |
| Gaia | DR3 | Gaia Collab. 2022 | Low-res XP spectra, photometry, astrometry |

All derivative checkpoints released on the Hugging Face Hub are distributed under an MIT licence; users are nevertheless responsible for complying with the upstream survey licences when redistributing raw data.

## Physical Units & Conventions

• **Images**: pixel values are calibrated nanomaggies.  Exposure time normalisation is survey-specific and automatically handled by the image codec.

• **Spectra**: flux density in erg s⁻¹ cm⁻² Å⁻¹ (observer frame).  Wavelengths are Å, *not* log-λ when inside the model.

• **Photometry / Scalars**: all fluxes in nanomaggies, magnitudes in the AB system.  Ellipticities use SDSS convention *(e₁,e₂)*.

## Known Limitations & Caveats

1. No ultraviolet (< 3500 Å) or mid-infrared (> 1 µm) spectral support.
2. HSC chip-edge artefacts occasionally propagate into synthetic spectra – crop images if necessary.
3. The model was trained on **96 × 96 px** cut-outs; objects extending beyond that FOV will be truncated.

## Citation

If you use AION-1 in a publication, please cite both the codebase and the accompanying paper:

```bibtex
@article{Francois2025aion,
  title       = {AION-1: Omnimodal Foundation Model for Astronomical Sciences},
  author      = {LASTNAME, Firstname et al.},
  journal     = {arXiv e-prints},
  year        = 2025,
  archivePrefix = {arXiv},
  eprint      = {2406.00000}
}
```

## Summary

AION-1's architecture represents a significant advance in multimodal scientific machine learning:

1. **Universal tokenization** handles arbitrary astronomical data types
2. **Unified transformer** learns cross-modal relationships
3. **Flexible design** adapts to available observations
4. **Emergent understanding** discovers physical relationships

This architecture provides a foundation for next-generation astronomical analysis, enabling scientists to leverage all available data for their research.
