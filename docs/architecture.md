# Code Architecture

This page explains the major components of the AION codebase and how they interact.

## Modality Data Classes

`aion/modalities.py` defines Pydantic models describing each input modality. Examples include `Image` for imaging data, `Spectrum` for spectroscopic data, and scalar modalities such as `FluxG` or `Parallax`. These classes provide type checked containers for the raw astronomy data.

## Codecs (Tokenizers)

Under `aion/codecs/` reside modality specific **Codecs**. A codec encodes a `Modality` instance into a sequence of discrete tokens and can decode tokens back to the original data. The base interface is defined in `codecs/base.py` and concrete implementations exist for images, spectra and catalog entries.

```python
from aion.codecs import ImageCodec
from aion.modalities import Image

image = Image(flux=my_flux, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])
codec = ImageCodec.from_pretrained("polymathic-ai/aion-image-codec")
tokens = codec.encode(image)
```

## FourM Architecture

The core transformer architecture lives in the `aion/fourm/` package. The `FourM` class combines encoder and decoder blocks along with modality embeddings. It provides utilities to concatenate tokens from different modalities and to apply modality-aware attention masks.

## AION Wrapper

`aion/model.py` defines the `AION` class which inherits from `FourM`. It adds high level helpers for:

- **`embed_inputs`** – convert a dictionary of modality tensors into encoder tokens.
- **`embed_targets`** – build decoder inputs and target masks for selected modalities.
- **`forward`** – run the full model returning logits for the requested targets.

Typical usage during inference is:

```python
from aion import AION

model = AION.from_pretrained("aion-base")
logits = model(input_dict, target_mask)
```

Here `input_dict` maps modality names to token tensors (obtained via the codecs) and `target_mask` selects which tokens to predict.

For additional details see the docstrings in `model.py` and the modules within `aion/fourm`.
