# AION-1 Usage Guide

This comprehensive guide demonstrates how to use AION-1 for various astronomical analysis tasks. From basic inference to advanced multimodal generation, you'll learn to leverage AION-1's capabilities for your research.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Loading and Preprocessing Data](#loading-and-preprocessing-data)
3. [Basic Inference](#basic-inference)
4. [Multimodal Generation](#multimodal-generation)
5. [Cross-Modal Translation](#cross-modal-translation)
6. [Representation Learning](#representation-learning)
7. [Advanced Applications](#advanced-applications)
8. [Performance Optimization](#performance-optimization)

## Quick Start

Let's begin with a simple example that showcases AION-1's core capabilities:

```python
import torch, numpy as np
from aion import AION
from aion.modalities import Image

# 1) Load a checkpoint (300 M parameters)
model = AION.from_pretrained('polymathic-ai/aion-tiny').eval()

# 2) Read an example 5-band HSC cut-out (units: nanomaggies)
flux_cube = np.load('hsc_cutout_5band.npy')  # shape (5,96,96)
img = Image(flux=flux_cube, bands=['HSC-G','HSC-R','HSC-I','HSC-Z','HSC-Y'])

# 3) Predict an SDSS-like spectrum (observer-frame, erg s⁻¹ cm⁻² Å⁻¹)
with torch.inference_mode():
    result = model.generate(inputs={'image': img}, targets=['spectrum'])

spec = result['spectrum']
print(f"Generated spectrum: λ range {spec.wavelength[0]:.0f}-{spec.wavelength[-1]:.0f} Å, shape={spec.flux.shape}")
```

## Loading and Preprocessing Data

### Working with Images

AION-1 expects images in a specific format. Here's how to prepare astronomical images:

```python
import numpy as np
from astropy.io import fits
from aion.modalities import Image
from aion.codecs.preprocessing import ImagePreprocessor

# Load FITS data
with fits.open('galaxy.fits') as hdul:
    # Assuming multi-band data in extensions
    flux_data = np.array([hdul[i].data for i in range(1, 6)])  # 5 bands

# Create Image modality
image = Image(
    flux=flux_data,
    bands=['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y'],
    # Optional: provide inverse variance for optimal processing
    ivar=inverse_variance_data
)

# Apply survey-specific preprocessing
preprocessor = ImagePreprocessor(survey='HSC')
processed_image = preprocessor(image)
```

### Working with Spectra

Load and prepare spectroscopic data:

```python
from aion.modalities import Spectrum
from astropy.io import fits

# Load SDSS spectrum
hdul = fits.open('spec-plate-mjd-fiber.fits')
wavelength = 10**hdul[1].data['loglam']  # Convert log wavelength
flux = hdul[1].data['flux']
ivar = hdul[1].data['ivar']

# Create Spectrum modality
spectrum = Spectrum(
    wavelength=wavelength,
    flux=flux,
    ivar=ivar,
    survey='SDSS'
)

# The model handles resampling to internal wavelength grid automatically
```

### Working with Catalog Data

Process tabular astronomical measurements:

```python
from aion.modalities import (
    FluxG, FluxR, FluxI, FluxZ,
    E1, E2, RadiusCARP, Redshift
)

# Load catalog data (e.g., from pandas DataFrame)
catalog_entry = {
    'flux_g': FluxG(value=catalog_df['flux_g'].values),
    'flux_r': FluxR(value=catalog_df['flux_r'].values),
    'e1': E1(value=catalog_df['e1'].values),
    'e2': E2(value=catalog_df['e2'].values),
    'radius': RadiusCARP(value=catalog_df['radius'].values)
}
```

## Basic Inference

### Single Modality Prediction

Predict missing photometric measurements from available data:

```python
# Given g,r,i bands, predict z band
inputs = {
    'flux_g': FluxG(value=[19.5]),
    'flux_r': FluxR(value=[18.2]),
    'flux_i': FluxI(value=[17.8])
}

# Predict z-band flux
with torch.no_grad():
    predictions = model.generate(
        inputs=inputs,
        targets=['flux_z']
    )

z_flux = predictions['flux_z'].value[0]
print(f"Predicted z-band flux: {z_flux:.2f}")
```

### Batch Processing

Process multiple objects efficiently:

```python
# Prepare batch of galaxies
batch_images = [load_galaxy(i) for i in range(32)]
batch = {
    'image': Image.batch(batch_images)
}

# Generate properties for all galaxies
with torch.no_grad():
    results = model.generate(
        inputs=batch,
        targets=['redshift', 'e1', 'e2', 'radius']
    )

# Extract results
redshifts = results['redshift'].value
ellipticities = np.sqrt(results['e1'].value**2 + results['e2'].value**2)
```

## Multimodal Generation

### Conditional Generation

Generate multiple modalities conditioned on partial observations:

```python
# Complex multimodal generation example
def analyze_galaxy(image_path, known_redshift=None):
    # Load image
    image = load_and_preprocess_image(image_path)

    inputs = {'image': image}
    if known_redshift:
        inputs['redshift'] = Redshift(value=[known_redshift])

    # Generate comprehensive analysis
    targets = [
        'spectrum',           # Full spectrum
        'flux_g', 'flux_r', 'flux_i', 'flux_z',  # Photometry
        'e1', 'e2',          # Shape parameters
        'radius',            # Size
        'parallax',          # Distance indicator
        'extinction_v'       # Dust extinction
    ]

    with torch.no_grad():
        results = model.generate(
            inputs=inputs,
            targets=targets,
            num_generations=1,
            temperature=1.0
        )

    return results

# Analyze a galaxy
galaxy_properties = analyze_galaxy('ngc1234.fits', known_redshift=0.05)
```

### Uncertainty Quantification

Generate multiple samples to estimate uncertainties:

```python
def estimate_uncertainty(inputs, target, num_samples=100):
    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            result = model.generate(
                inputs=inputs,
                targets=[target],
                temperature=1.2  # Higher temperature for more diversity
            )
            samples.append(result[target].value[0])

    samples = np.array(samples)
    return {
        'mean': np.mean(samples),
        'std': np.std(samples),
        'percentiles': np.percentile(samples, [16, 50, 84])
    }

# Estimate redshift uncertainty
z_stats = estimate_uncertainty(
    inputs={'image': galaxy_image},
    target='redshift'
)
print(f"Redshift: {z_stats['mean']:.3f} ± {z_stats['std']:.3f}")
```

## Cross-Modal Translation

### Image to Spectrum

Convert imaging observations to spectroscopic predictions:

```python
def image_to_spectrum(image, wavelength_range=(3800, 9200)):
    """Generate spectrum from multi-band image."""

    # Generate spectrum tokens
    with torch.no_grad():
        result = model.generate(
            inputs={'image': image},
            targets=['spectrum']
        )

    spectrum = result['spectrum']

    # Filter to desired wavelength range
    mask = (spectrum.wavelength >= wavelength_range[0]) & \
           (spectrum.wavelength <= wavelength_range[1])

    return {
        'wavelength': spectrum.wavelength[mask],
        'flux': spectrum.flux[mask]
    }

# Generate and plot spectrum
synthetic_spec = image_to_spectrum(galaxy_image)
plt.plot(synthetic_spec['wavelength'], synthetic_spec['flux'])
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux')
plt.title('AION-1 Generated Spectrum from Image')
```

### Spectrum to Image

Inverse translation - generate images from spectra:

```python
def spectrum_to_image(spectrum, bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']):
    """Generate multi-band image from spectrum."""

    with torch.no_grad():
        result = model.generate(
            inputs={'spectrum': spectrum},
            targets=['image'],
            target_bands=bands
        )

    return result['image']

# Reconstruct galaxy appearance
reconstructed_image = spectrum_to_image(observed_spectrum)
```

### Super-Resolution

Enhance low-resolution spectra using multimodal context:

```python
def enhance_spectrum(low_res_spectrum, supporting_data=None):
    """Enhance spectrum resolution using additional modalities."""

    inputs = {'spectrum': low_res_spectrum}

    # Add supporting data if available
    if supporting_data:
        inputs.update(supporting_data)

    # Generate high-resolution version
    with torch.no_grad():
        result = model.generate(
            inputs=inputs,
            targets=['spectrum_highres'],
            num_generations=1
        )

    return result['spectrum_highres']

# Example with photometric support
enhanced = enhance_spectrum(
    sdss_spectrum,
    supporting_data={
        'flux_g': FluxG(value=[18.5]),
        'flux_r': FluxR(value=[17.2])
    }
)
```

## Representation Learning

### Extracting Embeddings

Use AION-1's learned representations for downstream tasks:

```python
def extract_embeddings(data_dict, pool='mean'):
    """Extract feature embeddings from AION-1 encoder."""

    # Tokenize inputs
    tokens = model.tokenize(data_dict)

    # Get encoder representations
    with torch.no_grad():
        embeddings = model.encode(tokens)

    # Pool over sequence dimension
    if pool == 'mean':
        features = embeddings.mean(dim=1)
    elif pool == 'cls':
        features = embeddings[:, 0]  # First token
    elif pool == 'max':
        features = embeddings.max(dim=1)[0]

    return features.cpu().numpy()

# Extract features for clustering
galaxy_features = extract_embeddings({
    'image': galaxy_image,
    'spectrum': galaxy_spectrum
})
```

### Similarity Search

Find similar objects using learned representations:

```python
from sklearn.metrics.pairwise import cosine_similarity

class GalaxySimilaritySearch:
    def __init__(self, model):
        self.model = model
        self.database = []
        self.embeddings = []

    def add_galaxy(self, galaxy_data, metadata=None):
        """Add galaxy to search database."""
        embedding = extract_embeddings(galaxy_data)
        self.embeddings.append(embedding)
        self.database.append({
            'data': galaxy_data,
            'metadata': metadata,
            'embedding': embedding
        })

    def find_similar(self, query_data, k=10):
        """Find k most similar galaxies."""
        query_embedding = extract_embeddings(query_data)

        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            np.vstack(self.embeddings)
        )[0]

        # Get top k
        indices = np.argsort(similarities)[::-1][:k]

        return [(self.database[i], similarities[i]) for i in indices]

# Usage
searcher = GalaxySimilaritySearch(model)
# ... add galaxies to database ...
similar_galaxies = searcher.find_similar(query_galaxy, k=5)
```

### Anomaly Detection

Identify unusual objects using reconstruction error:

```python
def detect_anomalies(galaxies, threshold_percentile=95):
    """Detect anomalous galaxies using reconstruction error."""

    reconstruction_errors = []

    for galaxy in galaxies:
        # Encode and decode
        with torch.no_grad():
            reconstructed = model.generate(
                inputs=galaxy,
                targets=list(galaxy.keys())
            )

        # Compute reconstruction error
        error = 0
        for key in galaxy:
            if key == 'image':
                error += np.mean((galaxy[key].flux -
                                reconstructed[key].flux)**2)
            elif hasattr(galaxy[key], 'value'):
                error += np.mean((galaxy[key].value -
                                reconstructed[key].value)**2)

        reconstruction_errors.append(error)

    # Set threshold
    threshold = np.percentile(reconstruction_errors, threshold_percentile)

    # Identify anomalies
    anomalies = [g for g, e in zip(galaxies, reconstruction_errors)
                 if e > threshold]

    return anomalies, reconstruction_errors
```

## Advanced Applications

### Multi-Survey Integration

Combine observations from different surveys:

```python
def integrate_multi_survey(hsc_image, sdss_spectrum, desi_spectrum=None):
    """Integrate observations from multiple surveys."""

    inputs = {
        'image': hsc_image,
        'spectrum_sdss': sdss_spectrum
    }

    if desi_spectrum:
        inputs['spectrum_desi'] = desi_spectrum

    # Generate unified representation
    with torch.no_grad():
        # Extract all available properties
        results = model.generate(
            inputs=inputs,
            targets=['redshift', 'stellar_mass', 'sfr', 'metallicity']
        )

        # Generate missing modalities
        if not desi_spectrum:
            results['spectrum_desi'] = model.generate(
                inputs=inputs,
                targets=['spectrum_desi']
            )['spectrum_desi']

    return results
```

### Time Series Analysis

Analyze variable objects across epochs:

```python
def analyze_variable_object(observations):
    """
    Analyze time-variable astronomical object.

    observations: list of (time, data_dict) tuples
    """

    embeddings_over_time = []
    properties_over_time = []

    for time, data in observations:
        # Extract embeddings
        embedding = extract_embeddings(data)
        embeddings_over_time.append(embedding)

        # Predict properties
        with torch.no_grad():
            props = model.generate(
                inputs=data,
                targets=['flux_g', 'flux_r', 'temperature']
            )

        properties_over_time.append({
            'time': time,
            'properties': props,
            'embedding': embedding
        })

    # Analyze evolution
    embeddings = np.vstack(embeddings_over_time)

    # Detect significant changes
    embedding_distances = np.sqrt(np.sum(np.diff(embeddings, axis=0)**2, axis=1))
    change_points = np.where(embedding_distances > np.std(embedding_distances) * 2)[0]

    return {
        'properties': properties_over_time,
        'change_points': change_points,
        'embedding_evolution': embeddings
    }
```

### Physical Parameter Estimation

Estimate astrophysical parameters with uncertainty:

```python
class PhysicalParameterEstimator:
    def __init__(self, model, num_samples=100):
        self.model = model
        self.num_samples = num_samples

    def estimate_parameters(self, observations):
        """Estimate physical parameters with uncertainties."""

        # Parameters to estimate
        parameters = [
            'redshift', 'stellar_mass', 'sfr',
            'metallicity', 'age', 'extinction_v'
        ]

        # Generate multiple samples
        samples = {param: [] for param in parameters}

        with torch.no_grad():
            for _ in range(self.num_samples):
                results = self.model.generate(
                    inputs=observations,
                    targets=parameters,
                    temperature=1.1
                )

                for param in parameters:
                    if param in results:
                        samples[param].append(results[param].value[0])

        # Compute statistics
        estimates = {}
        for param, values in samples.items():
            if values:
                values = np.array(values)
                estimates[param] = {
                    'median': np.median(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'ci_68': np.percentile(values, [16, 84]),
                    'ci_95': np.percentile(values, [2.5, 97.5])
                }

        return estimates

# Usage
estimator = PhysicalParameterEstimator(model)
parameters = estimator.estimate_parameters({
    'image': galaxy_image,
    'spectrum': galaxy_spectrum
})

print(f"Stellar Mass: {parameters['stellar_mass']['median']:.2e} "
      f"+/- {parameters['stellar_mass']['std']:.2e} M_sun")
```

## Performance Optimization

### Efficient Batch Processing

```python
from torch.utils.data import DataLoader, Dataset

class AIONDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def process_large_dataset(data_list, batch_size=32):
    """Efficiently process large datasets."""

    dataset = AIONDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                          num_workers=4, pin_memory=True)

    all_results = []

    with torch.no_grad():
        for batch in dataloader:
            # Process batch
            results = model.generate(
                inputs=batch,
                targets=['redshift', 'stellar_mass']
            )
            all_results.append(results)

    # Concatenate results
    return {k: np.concatenate([r[k].value for r in all_results])
            for k in all_results[0]}
```

### Memory-Efficient Processing

```python
def process_with_chunking(large_spectrum, chunk_size=1000):
    """Process very long spectra in chunks."""

    n_chunks = len(large_spectrum.wavelength) // chunk_size + 1
    chunk_results = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(large_spectrum.wavelength))

        chunk = Spectrum(
            wavelength=large_spectrum.wavelength[start:end],
            flux=large_spectrum.flux[start:end]
        )

        with torch.no_grad():
            result = model.process_spectrum_chunk(chunk)
            chunk_results.append(result)

    # Combine chunks
    return combine_spectrum_chunks(chunk_results)
```

### GPU Memory Management

```python
import gc

def memory_efficient_generation(inputs, targets, max_batch=16):
    """Generate with automatic batch size adjustment."""

    batch_size = max_batch

    while batch_size > 0:
        try:
            with torch.no_grad():
                results = model.generate(
                    inputs=inputs,
                    targets=targets,
                    batch_size=batch_size
                )
            return results

        except torch.cuda.OutOfMemoryError:
            # Clear cache and try smaller batch
            torch.cuda.empty_cache()
            gc.collect()
            batch_size //= 2

            if batch_size == 0:
                raise RuntimeError("Cannot fit even batch size 1")

    raise RuntimeError("Failed to process")
```

## Best Practices

### 1. Data Preparation
- Always normalize and preprocess data according to survey specifications
- Provide inverse variance when available for optimal results
- Use appropriate data types for each modality

### 2. Model Selection
- Use `aion-tiny` for quick experiments and limited GPU memory
- Use `aion-base` for most research applications
- Use `aion-large` for highest accuracy when computational resources permit

### 3. Generation Settings
- Lower temperature (0.8-1.0) for more deterministic outputs
- Higher temperature (1.1-1.5) for diversity and uncertainty estimation
- Multiple generations for robust uncertainty quantification

### 4. Error Handling
```python
def safe_generate(model, inputs, targets, fallback=None):
    """Safely generate with error handling."""
    try:
        return model.generate(inputs=inputs, targets=targets)
    except Exception as e:
        print(f"Generation failed: {e}")
        return fallback or {t: None for t in targets}
```

## Conclusion

AION-1 provides a powerful and flexible framework for multimodal astronomical analysis. Its ability to seamlessly integrate diverse observations enables new research possibilities:

- Cross-modal prediction and generation
- Unified analysis across multiple surveys
- Robust uncertainty quantification
- Discovery of unusual objects
- Efficient processing of large datasets

For more examples and the latest updates, visit the [AION GitHub repository](https://github.com/polymathic-ai/aion) and join our community discussions.
