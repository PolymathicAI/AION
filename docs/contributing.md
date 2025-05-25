# Contributing to AION-1

Welcome to the AION-1 project! We're thrilled that you're interested in contributing to the first large-scale multimodal foundation model for astronomy. This guide will help you get started with contributing, whether you're fixing bugs, adding features, improving documentation, or conducting research with AION-1.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Contribution Types](#contribution-types)
4. [Code Standards](#code-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.10 or later
- Git for version control
- CUDA-capable GPU (recommended for testing)
- Familiarity with PyTorch and transformers

### Understanding AION-1

Before diving into code, we recommend:

1. Reading the [AION-1 paper](https://arxiv.org/abs/XXXX.XXXXX)
2. Exploring the [Architecture documentation](architecture.html)
3. Running through the [Usage examples](usage.html)
4. Joining our [Discord community](https://discord.gg/polymathic-ai)

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/aion.git
cd aion
git remote add upstream https://github.com/polymathic-ai/aion.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv-dev
source venv-dev/bin/activate  # On Windows: venv-dev\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### 3. Download Test Data

```bash
# Download minimal test datasets
python scripts/download_test_data.py

# Verify installation
python -m pytest tests/test_installation.py
```

## Contribution Types

### 🐛 Bug Fixes

Found a bug? Here's how to fix it:

1. **Check existing issues** to avoid duplicates
2. **Create a minimal reproduction** script
3. **Write a test** that fails with the bug
4. **Fix the bug** and ensure the test passes
5. **Submit a PR** with clear description

Example bug fix workflow:
```python
# tests/test_bug_fix.py
def test_spectrum_interpolation_edge_case():
    """Test for issue #123: spectrum interpolation fails at boundaries."""
    spectrum = Spectrum(
        wavelength=np.array([3500, 3501, 10400]),
        flux=np.array([1.0, 1.5, 2.0])
    )

    # This should not raise an exception
    interpolated = spectrum.resample(np.linspace(3500, 10400, 100))
    assert len(interpolated.flux) == 100
```

### ✨ New Features

Adding new capabilities to AION-1:

1. **Discuss first**: Open an issue or discussion
2. **Design document**: For major features, write a brief design doc
3. **Implement incrementally**: Break into small PRs
4. **Add tests and docs**: Every feature needs both

#### Adding a New Modality

Here's an example of adding a new modality:

```python
# aion/modalities.py
class TimeSeries(Modality):
    """
    Time series astronomical measurements.

    Attributes:
        time: Time stamps in MJD
        flux: Flux measurements
        error: Measurement uncertainties
    """
    time: np.ndarray
    flux: np.ndarray
    error: Optional[np.ndarray] = None

    def validate(self):
        """Ensure time series is properly formatted."""
        assert len(self.time) == len(self.flux)
        assert np.all(np.diff(self.time) >= 0), "Time must be monotonic"

# aion/codecs/timeseries.py
class TimeSeriesCodec(Codec):
    """Tokenizer for astronomical time series."""

    def encode(self, timeseries: TimeSeries) -> torch.Tensor:
        # Implementation here
        pass

    def decode(self, tokens: torch.Tensor) -> TimeSeries:
        # Implementation here
        pass
```

### 📚 Documentation Improvements

Good documentation is crucial:

- **Fix typos and clarify**: Even small improvements help
- **Add examples**: Real-world usage examples
- **Improve API docs**: Better docstrings
- **Write tutorials**: Step-by-step guides

### 🔬 Research Contributions

Using AION-1 for research? Consider contributing:

- **Benchmarks**: Performance on astronomical tasks
- **Fine-tuning scripts**: For specific applications
- **Analysis notebooks**: Demonstrating capabilities
- **Model improvements**: Better architectures or training

## Code Standards

### Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Clear variable names and type hints
def process_galaxy_spectrum(
    spectrum: Spectrum,
    redshift: float,
    extinction_curve: Optional[np.ndarray] = None
) -> Spectrum:
    """
    Process galaxy spectrum with redshift and extinction corrections.

    Args:
        spectrum: Input spectrum
        redshift: Cosmological redshift
        extinction_curve: Optional extinction curve

    Returns:
        Corrected spectrum
    """
    # De-redshift
    corrected_wavelength = spectrum.wavelength / (1 + redshift)

    # Apply extinction if provided
    if extinction_curve is not None:
        extinction_factor = np.interp(
            corrected_wavelength,
            EXTINCTION_WAVELENGTH,
            extinction_curve
        )
        corrected_flux = spectrum.flux * extinction_factor
    else:
        corrected_flux = spectrum.flux

    return Spectrum(
        wavelength=corrected_wavelength,
        flux=corrected_flux,
        ivar=spectrum.ivar
    )
```

### Type Hints

Always use type hints for better code clarity:

```python
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

def tokenize_multimodal(
    data: Dict[str, Modality],
    codecs: Dict[str, Codec],
    max_length: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Tokenize multiple modalities."""
    tokens = {}
    for modality_name, modality_data in data.items():
        if modality_name in codecs:
            tokens[modality_name] = codecs[modality_name].encode(modality_data)
    return tokens
```

### Docstrings

Use Google-style docstrings:

```python
def cross_match_catalogs(
    catalog1: Catalog,
    catalog2: Catalog,
    radius: float = 1.0,
    unit: str = 'arcsec'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-match two astronomical catalogs.

    Performs positional cross-matching between two catalogs using
    a specified search radius.

    Args:
        catalog1: First catalog
        catalog2: Second catalog
        radius: Search radius for matching
        unit: Unit of radius ('arcsec', 'arcmin', 'deg')

    Returns:
        Tuple containing:
            - indices1: Matched indices from catalog1
            - indices2: Matched indices from catalog2
            - distances: Angular distances of matches

    Raises:
        ValueError: If unit is not recognized

    Example:
        >>> idx1, idx2, dist = cross_match_catalogs(
        ...     gaia_catalog,
        ...     sdss_catalog,
        ...     radius=2.0
        ... )
    """
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Fast unit tests
├── integration/       # Integration tests
├── fixtures/         # Test data and fixtures
└── benchmarks/       # Performance benchmarks
```

### Writing Tests

```python
# tests/unit/test_spectrum_codec.py
import pytest
import numpy as np
from aion.modalities import Spectrum
from aion.codecs import SpectrumCodec

class TestSpectrumCodec:
    @pytest.fixture
    def sample_spectrum(self):
        """Create a sample spectrum for testing."""
        wavelength = np.linspace(4000, 8000, 1000)
        flux = np.random.randn(1000) + 10
        return Spectrum(wavelength=wavelength, flux=flux)

    @pytest.fixture
    def codec(self):
        """Initialize spectrum codec."""
        return SpectrumCodec(
            latent_wavelength=np.linspace(3500, 10500, 8704)
        )

    def test_encode_decode_preserves_shape(self, sample_spectrum, codec):
        """Test that encode/decode preserves spectrum shape."""
        tokens = codec.encode(sample_spectrum)
        reconstructed = codec.decode(tokens)

        assert reconstructed.wavelength.shape == sample_spectrum.wavelength.shape
        assert reconstructed.flux.shape == sample_spectrum.flux.shape

    def test_handles_missing_data(self, codec):
        """Test codec handles spectra with gaps."""
        wavelength = np.array([4000, 4100, 4200, 6000, 6100])
        flux = np.array([1.0, 1.1, 1.2, 2.0, 2.1])

        spectrum = Spectrum(wavelength=wavelength, flux=flux)
        tokens = codec.encode(spectrum)

        assert tokens is not None
        assert len(tokens.shape) == 2  # [batch, seq_len]
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_spectrum_codec.py

# Run with coverage
pytest --cov=aion --cov-report=html

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Documentation

### Building Documentation

```bash
cd docs
make html
# View at docs/_build/html/index.html
```

### Writing Documentation

When adding new features, update:

1. **Docstrings**: In the code itself
2. **API Reference**: In `docs/api.md`
3. **Usage Examples**: In `docs/usage.md`
4. **Architecture**: If design changes

Example documentation addition:

```markdown
### Working with Time Series

AION-1 can process variable star light curves and other time series data:

\```python
from aion.modalities import TimeSeries

# Load light curve data
lightcurve = TimeSeries(
    time=mjd_times,
    flux=flux_measurements,
    error=flux_errors
)

# Generate period estimate
results = model.generate(
    inputs={'timeseries': lightcurve},
    targets=['period', 'variability_class']
)

print(f"Period: {results['period'].value[0]:.3f} days")
print(f"Class: {results['variability_class'].value[0]}")
\```
```

## Submitting Changes

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Guidelines

Use conventional commits:

```bash
# Format: <type>(<scope>): <subject>

git commit -m "feat(modalities): add time series support"
git commit -m "fix(codec): handle edge case in spectrum interpolation"
git commit -m "docs(api): improve codec documentation"
git commit -m "test(integration): add multi-survey processing tests"
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Clear title**: Summarize the change
- **Description**: Explain what and why
- **Tests**: Confirm all tests pass
- **Screenshots**: If relevant (e.g., for visualizations)

### 5. Code Review

- Respond to feedback constructively
- Make requested changes
- Keep PR focused and reasonably sized

## Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Key points:

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive criticism
- Report unacceptable behavior

### Getting Help

- **Discord**: Quick questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Longer form conversations
- **Office Hours**: Weekly community calls (Thursdays 3pm UTC)

### Recognition

Contributors are recognized in:

- The `CONTRIBUTORS.md` file
- Release notes
- Research papers (for significant contributions)

## Advanced Topics

### Adding New Surveys

To add support for a new astronomical survey:

1. **Define band mappings** in `aion/surveys.py`
2. **Add preprocessing** in `aion/codecs/preprocessing/`
3. **Update documentation** with survey details
4. **Add tests** with sample data

### Performance Optimization

When optimizing AION-1:

```python
# Profile first
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = model.generate(inputs, targets)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def process_large_batch(model, data):
    # Function to profile
    pass
```

## Thank You!

Your contributions make AION-1 better for the entire astronomical community. Whether you're fixing a typo, adding a feature, or conducting research, every contribution matters.

If you have questions or need help getting started, don't hesitate to reach out on Discord or open an issue. We're here to help!

Happy contributing! 🌟🔭
