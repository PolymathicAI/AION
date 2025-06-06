# API Reference

This comprehensive API reference covers all major components of AION-1, including modalities, codecs, models, and utilities.

## Core Model

### `aion.AION`

The main AION model class that provides high-level interfaces for multimodal astronomical analysis.

```python
class AION(FourM):
    """
    AION-1 multimodal astronomical foundation model.

    Inherits from FourM architecture and adds astronomical-specific
    functionality for processing 39 different data modalities.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.float32,
        **kwargs
    ) -> 'AION':
        """
        Load a pre-trained AION model.

        Args:
            model_name: HuggingFace model identifier
                - 'polymathic-ai/aion-tiny': 300M parameter model
                - 'polymathic-ai/aion-base': 800M parameter model
                - 'polymathic-ai/aion-large': 3.1B parameter model
            device: Device to load model on ('cuda', 'cpu', 'mps')
            torch_dtype: Data type for model weights
            **kwargs: Additional arguments passed to model constructor

        Returns:
            AION model instance
        """

    def generate(
        self,
        inputs: Dict[str, Modality],
        targets: List[str],
        num_generations: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Modality]:
        """
        Generate target modalities from input observations.

        Note:
            ``targets`` must be chosen from the list returned by
            ``AION.supported_targets`` (essentially the 39 modality names
            listed in the architecture documentation).  Supplying an
            unsupported string will raise ``ValueError``.

        Args:
            inputs: Dictionary mapping modality names to data
            targets: List of modality names to generate
            num_generations: Number of samples to generate
            temperature: Sampling temperature (higher = more diverse)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Dictionary mapping target names to generated modalities
        """

    def encode(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode input tokens to learned representations.

        Args:
            inputs: Tokenized inputs

        Returns:
            Encoder hidden states [batch, seq_len, hidden_dim]
        """

    def tokenize(
        self,
        modalities: Dict[str, Modality]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert modalities to discrete tokens using codecs.

        Args:
            modalities: Dictionary of modality data

        Returns:
            Dictionary of tokenized tensors
        """
```

## Modalities

AION-1 supports 39 different astronomical data modalities. Each modality is represented by a Pydantic model ensuring type safety and validation.

### Image Modalities

#### `aion.modalities.Image`

```python
class Image(Modality):
    """
    Multi-band astronomical image.

    Attributes:
        flux: Image data array [bands, height, width]
        bands: List of band identifiers (e.g., ['HSC-G', 'HSC-R'])
        ivar: Optional inverse variance array for weighting
        mask: Optional boolean mask array
    """

    flux: np.ndarray
    bands: List[str]
    ivar: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    @classmethod
    def batch(cls, images: List['Image']) -> 'Image':
        """Batch multiple images together."""

    def crop(self, size: int = 96) -> 'Image':
        """Center crop image to specified size."""
```

### Spectrum Modalities

#### `aion.modalities.Spectrum`

```python
class Spectrum(Modality):
    """
    Astronomical spectrum.

    Attributes:
        wavelength: Wavelength array in Angstroms
        flux: Flux density array
        ivar: Optional inverse variance
        survey: Source survey identifier
    """

    wavelength: np.ndarray
    flux: np.ndarray
    ivar: Optional[np.ndarray] = None
    survey: Optional[str] = None

    def resample(
        self,
        new_wavelength: np.ndarray
    ) -> 'Spectrum':
        """Resample spectrum to new wavelength grid."""

    def normalize(self) -> 'Spectrum':
        """Apply median normalization."""
```

### Scalar Modalities

AION-1 includes numerous scalar modalities for photometry, shapes, and physical parameters:

#### Photometric Fluxes

```python
class FluxG(ScalarModality):
    """g-band flux measurement."""
    value: np.ndarray
    error: Optional[np.ndarray] = None

class FluxR(ScalarModality):
    """r-band flux measurement."""
    value: np.ndarray
    error: Optional[np.ndarray] = None

class FluxI(ScalarModality):
    """i-band flux measurement."""
    value: np.ndarray
    error: Optional[np.ndarray] = None

class FluxZ(ScalarModality):
    """z-band flux measurement."""
    value: np.ndarray
    error: Optional[np.ndarray] = None
```

#### Shape Parameters

```python
class E1(ScalarModality):
    """First ellipticity component."""
    value: np.ndarray

class E2(ScalarModality):
    """Second ellipticity component."""
    value: np.ndarray

class RadiusCARP(ScalarModality):
    """CARP radius measurement."""
    value: np.ndarray
```

#### Physical Properties

```python
class Redshift(ScalarModality):
    """Spectroscopic or photometric redshift."""
    value: np.ndarray
    error: Optional[np.ndarray] = None

class ExtinctionV(ScalarModality):
    """V-band extinction."""
    value: np.ndarray

class Parallax(ScalarModality):
    """Parallax measurement in mas."""
    value: np.ndarray
    error: Optional[np.ndarray] = None
```

### Catalog Modalities

#### `aion.modalities.Catalog`

```python
class Catalog(Modality):
    """
    Astronomical object catalog.

    Attributes:
        entries: List of catalog objects
        max_objects: Maximum number of objects to process
    """

    entries: List[CatalogEntry]
    max_objects: int = 100

    def sort_by_distance(self) -> 'Catalog':
        """Sort entries by distance from center."""

    def filter_bright(self, magnitude_limit: float) -> 'Catalog':
        """Filter to objects brighter than limit."""
```

## Codecs (Tokenizers)

Codecs convert between modalities and discrete tokens. Each modality type has a specialized codec.

### Base Codec Interface

#### `aion.codecs.base.Codec`

```python
class Codec(ABC):
    """
    Abstract base class for modality codecs.
    """

    @abstractmethod
    def encode(self, modality: Modality) -> torch.Tensor:
        """Encode modality to discrete tokens."""

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> Modality:
        """Decode tokens back to modality."""

    @classmethod
    def from_pretrained(cls, path: str) -> 'Codec':
        """Load pre-trained codec."""

    def save_pretrained(self, path: str):
        """Save codec weights and configuration."""
```

### Image Codec

#### `aion.codecs.ImageCodec`

```python
class ImageCodec(Codec):
    """
    Image tokenizer using MagVit architecture.

    Supports multi-survey images with different band counts
    through a unified channel embedding scheme.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_embed: int = 10000,
        compression_levels: int = 2,
        quantizer: str = 'fsq'
    ):
        """
        Initialize image codec.

        Args:
            hidden_dim: Hidden dimension size
            n_embed: Codebook size
            compression_levels: Spatial compression factor
            quantizer: Quantization method ('fsq' or 'vq')
        """

    def preprocess(
        self,
        image: Image,
        crop_size: int = 96
    ) -> torch.Tensor:
        """Apply survey-specific preprocessing."""

    def get_latent_shape(
        self,
        image_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Get shape of latent representation."""
```

### Spectrum Codec

#### `aion.codecs.SpectrumCodec`

```python
class SpectrumCodec(Codec):
    """
    Spectrum tokenizer using ConvNeXt V2 architecture.

    Uses a shared latent wavelength grid to handle spectra
    from different instruments.
    """

    def __init__(
        self,
        latent_wavelength: np.ndarray,
        hidden_dims: List[int] = [96, 192, 384, 768],
        n_embed: int = 1024,
        quantizer: str = 'lfq'
    ):
        """
        Initialize spectrum codec.

        Args:
            latent_wavelength: Target wavelength grid
            hidden_dims: ConvNeXt stage dimensions
            n_embed: Codebook size
            quantizer: Quantization method
        """

    def to_latent_grid(
        self,
        spectrum: Spectrum
    ) -> torch.Tensor:
        """Interpolate spectrum to latent wavelength grid."""
```

### Scalar Codec

#### `aion.codecs.ScalarCodec`

```python
class ScalarCodec(Codec):
    """
    Tokenizer for scalar quantities using adaptive quantization.
    """

    def __init__(
        self,
        quantizer_type: str = 'reservoir',
        n_bins: int = 256
    ):
        """
        Initialize scalar codec.

        Args:
            quantizer_type: Type of quantizer
                - 'linear': Uniform bins
                - 'log': Logarithmic bins
                - 'reservoir': Learned adaptive bins
                - 'compressed': Transform then quantize
            n_bins: Number of quantization levels
        """

    def fit(self, values: np.ndarray):
        """Fit quantizer to data distribution."""
```

## Quantizers

Quantization modules that convert continuous values to discrete tokens.

### `aion.codecs.quantizers.FSQ`

```python
class FiniteScalarQuantization(nn.Module):
    """
    Finite Scalar Quantization from MagVit.

    Factorizes codebook into multiple small codebooks for
    better gradient flow and training stability.
    """

    def __init__(
        self,
        levels: List[int] = [8, 5, 5, 5, 5],
        eps: float = 1e-3
    ):
        """
        Args:
            levels: Number of levels per dimension
            eps: Small constant for numerical stability
        """
```

### `aion.codecs.quantizers.LFQ`

```python
class LookupFreeQuantization(nn.Module):
    """
    Lookup-Free Quantization using entropy regularization.

    Achieves quantization without explicit codebook lookup,
    improving training efficiency.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        entropy_weight: float = 0.1
    ):
        """
        Args:
            dim: Embedding dimension
            codebook_size: Target vocabulary size
            entropy_weight: Entropy regularization weight
        """
```

## Preprocessing

Survey-specific preprocessing utilities.

### `aion.codecs.preprocessing.ImagePreprocessor`

```python
class ImagePreprocessor:
    """
    Survey-specific image preprocessing.
    """

    def __init__(self, survey: str):
        """
        Initialize for specific survey.

        Args:
            survey: Survey name ('HSC', 'DES', 'SDSS', etc.)
        """

    def __call__(self, image: Image) -> torch.Tensor:
        """Apply preprocessing pipeline."""

    def get_rescaling_params(self) -> Dict[str, float]:
        """Get survey-specific rescaling parameters."""
```

### `aion.codecs.preprocessing.SpectrumPreprocessor`

```python
class SpectrumPreprocessor:
    """
    Spectrum normalization and preprocessing.
    """

    def normalize_median(
        self,
        spectrum: Spectrum
    ) -> Spectrum:
        """Apply median normalization."""

    def mask_skylines(
        self,
        spectrum: Spectrum
    ) -> Spectrum:
        """Mask common sky emission lines."""
```

## Model Components

### `aion.fourm.FourM`

```python
class FourM(nn.Module):
    """
    Base multimodal transformer architecture.

    Implements the encoder-decoder architecture with
    modality-specific embeddings and flexible attention.
    """

    def __init__(
        self,
        encoder_depth: int = 12,
        decoder_depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_bias: bool = False
    ):
        """Initialize FourM architecture."""
```

### `aion.fourm.encoder_embeddings`

```python
class ModalityEmbedding(nn.Module):
    """
    Learnable embeddings for each modality type.

    Provides both modality identification and survey
    provenance information.
    """

    def __init__(
        self,
        num_modalities: int,
        num_surveys: int,
        embed_dim: int
    ):
        """Initialize modality embeddings."""
```

## Utilities

### `aion.model_utils`

```python
def load_codec(modality: str, device: str = 'cuda') -> Codec:
    """Load pre-trained codec for modality."""

def create_model_config(
    model_size: str = 'base'
) -> Dict[str, Any]:
    """Get configuration for model size."""

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
```

### `aion.generation_utils`

```python
def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Sample from logits with temperature scaling.

    Args:
        logits: Model output logits
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold

    Returns:
        Sampled token indices
    """

def generate_with_caching(
    model: AION,
    inputs: Dict[str, torch.Tensor],
    max_length: int,
    use_cache: bool = True
) -> torch.Tensor:
    """Generate tokens with KV caching for efficiency."""
```

## Data Loading

### `aion.data.AstronomicalDataset`

```python
class AstronomicalDataset(Dataset):
    """
    PyTorch dataset for astronomical observations.
    """

    def __init__(
        self,
        data_paths: List[str],
        modalities: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.

        Args:
            data_paths: Paths to data files
            modalities: List of modalities to load
            transform: Optional data transformation
        """

    def __getitem__(self, idx: int) -> Dict[str, Modality]:
        """Get single observation."""
```

## Example Usage

### Complete Pipeline

```python
import torch
from aion import AION
from aion.modalities import Image, Spectrum
from aion.codecs import ImageCodec, SpectrumCodec

# Load model and codecs
model = AION.from_pretrained('polymathic-ai/aion-base')
image_codec = ImageCodec.from_pretrained('polymathic-ai/aion-image-codec')
spectrum_codec = SpectrumCodec.from_pretrained('polymathic-ai/aion-spectrum-codec')

# Load data
image = Image(flux=galaxy_flux, bands=['g', 'r', 'i', 'z', 'y'])
spectrum = Spectrum(wavelength=wavelength, flux=flux)

# Tokenize
tokens = {
    'image': image_codec.encode(image),
    'spectrum': spectrum_codec.encode(spectrum)
}

# Encode to representations
with torch.no_grad():
    representations = model.encode(tokens)

# Generate missing modalities
results = model.generate(
    inputs={'image': image},
    targets=['spectrum', 'redshift']
)

# Decode results
generated_spectrum = spectrum_codec.decode(results['spectrum'])
print(f"Predicted redshift: {results['redshift'].value[0]:.3f}")
```

## Error Handling

All AION components include comprehensive error handling:

```python
from aion.exceptions import (
    ModalityError,      # Invalid modality data
    CodecError,         # Tokenization failures
    ModelError,         # Model inference errors
    DataError          # Data loading issues
)

try:
    result = model.generate(inputs, targets)
except ModalityError as e:
    print(f"Invalid modality: {e}")
except CodecError as e:
    print(f"Tokenization failed: {e}")
```

## Performance Tips

1. **Batch Processing**: Always process multiple objects together when possible
2. **Mixed Precision**: Use `torch.cuda.amp` for faster inference
3. **Token Caching**: Reuse encoder outputs when generating multiple targets
4. **Device Placement**: Use `.to(device)` consistently for all tensors

For more details, see the [Usage Guide](usage.md) and [Architecture](architecture.md) documentation.

```{eval-rst}
.. automodule:: aion
   :members:
   :undoc-members:
   :show-inheritance:
```
