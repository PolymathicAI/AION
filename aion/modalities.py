from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from jaxtyping import Bool, Float, Int
from torch import Tensor

__all__ = [
    "LegacySurveyImage",
    "HSCImage",
    "DESISpectrum",
    "SDSSSpectrum",
    "LegacySurveyCatalog",
    "LegacySurveySegmentationMap",
    "LegacySurveyFluxG",
    "LegacySurveyFluxR",
    "LegacySurveyFluxI",
    "LegacySurveyFluxZ",
    "LegacySurveyFluxW1",
    "LegacySurveyFluxW2",
    "LegacySurveyFluxW3",
    "LegacySurveyFluxW4",
    "LegacySurveyShapeR",
    "LegacySurveyShapeE1",
    "LegacySurveyShapeE2",
    "LegacySurveyEBV",
    "Z",
    "HSCAG",
    "HSCAR",
    "HSCAI",
    "HSCAZ",
    "HSCAY",
    "HSCMagG",
    "HSCMagR",
    "HSCMagI",
    "HSCMagZ",
    "HSCMagY",
    "HSCShape11",
    "HSCShape22",
    "HSCShape12",
    "GaiaFluxG",
    "GaiaFluxBp",
    "GaiaFluxRp",
    "GaiaParallax",
    "Ra",
    "Dec",
    "GaiaXpBp",
    "GaiaXpRp",
]


class BaseModality(ABC):
    """Base class for all modality data types."""


class Modality(BaseModality, ABC):
    """Base class for all token modalities."""

    token_key: ClassVar[str] = ""


@dataclass
class Image(BaseModality):
    """Base class for image modality data.

    This is an abstract base class. Use LegacySurveyImage or HSCImage instead.
    """

    flux: Float[Tensor, " batch num_bands height width"]
    bands: list[str]

    def __repr__(self) -> str:
        repr_str = f"Image(flux_shape={list(self.flux.shape)}, bands={self.bands})"
        return repr_str


class HSCImage(Image, Modality):
    """HSC image modality data."""

    token_key: ClassVar[str] = "tok_image_hsc"


class LegacySurveyImage(Image, Modality):
    """Legacy Survey image modality data."""

    token_key: ClassVar[str] = "tok_image"


@dataclass
class Spectrum(BaseModality):
    """Base class for spectrum modality data.

    This is an abstract base class. Use DESISpectrum or SDSSSpectrum instead.
    """

    flux: Float[Tensor, " batch length"]
    ivar: Float[Tensor, " batch length"]
    mask: Bool[Tensor, " batch length"]
    wavelength: Float[Tensor, " batch length"]

    def __repr__(self) -> str:
        repr_str = (
            f"Spectrum(flux_shape={list(self.flux.shape)}, "
            f"wavelength_range=[{self.wavelength.min().item():.1f}, "
            f"{self.wavelength.max().item():.1f}])"
        )
        return repr_str


class DESISpectrum(Spectrum, Modality):
    """DESI spectrum modality data."""

    token_key: ClassVar[str] = "tok_spectrum_desi"


class SDSSSpectrum(Spectrum, Modality):
    """SDSS spectrum modality data."""

    token_key: ClassVar[str] = "tok_spectrum_sdss"


# Catalog modality
@dataclass
class LegacySurveyCatalog(Modality):
    """Catalog modality data.

    Represents a catalog of scalar values from the Legacy Survey.
    """

    X: Int[Tensor, " batch n"]
    Y: Int[Tensor, " batch n"]
    SHAPE_E1: Float[Tensor, " batch n"]
    SHAPE_E2: Float[Tensor, " batch n"]
    SHAPE_R: Float[Tensor, " batch n"]
    token_key: ClassVar[str] = "catalog"


@dataclass
class LegacySurveySegmentationMap(Modality):
    """Legacy Survey segmentation map modality data.

    Represents 2D segmentation maps built from Legacy Survey detections.
    """

    field: Float[Tensor, " batch height width"]
    token_key: ClassVar[str] = "tok_segmap"

    def __repr__(self) -> str:
        repr_str = f"LegacySurveySegmentationMap(field_shape={list(self.field.shape)})"
        return repr_str


@dataclass
class Scalar(Modality):
    """Base class for scalar modality data.

    Represents a single scalar value per sample, typically used for
    flux measurements, shape parameters, or other single-valued properties.
    """

    value: Float[Tensor, "..."]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={list(self.value.shape)})"


# Flux measurements in different bands
class LegacySurveyFluxG(Scalar, Modality):
    """G-band flux measurement from Legacy Survey."""

    name: ClassVar[str] = "FLUX_G"
    token_key: ClassVar[str] = "tok_flux_g"


class LegacySurveyFluxR(Scalar, Modality):
    """R-band flux measurement."""

    name: ClassVar[str] = "FLUX_R"
    token_key: ClassVar[str] = "tok_flux_r"


class LegacySurveyFluxI(Scalar, Modality):
    """I-band flux measurement."""

    name: ClassVar[str] = "FLUX_I"
    token_key: ClassVar[str] = "tok_flux_i"


class LegacySurveyFluxZ(Scalar, Modality):
    """Z-band flux measurement."""

    name: ClassVar[str] = "FLUX_Z"
    token_key: ClassVar[str] = "tok_flux_z"


class LegacySurveyFluxW1(Scalar, Modality):
    """WISE W1-band flux measurement."""

    name: ClassVar[str] = "FLUX_W1"
    token_key: ClassVar[str] = "tok_flux_w1"


class LegacySurveyFluxW2(Scalar, Modality):
    """WISE W2-band flux measurement."""

    name: ClassVar[str] = "FLUX_W2"
    token_key: ClassVar[str] = "tok_flux_w2"


class LegacySurveyFluxW3(Scalar, Modality):
    """WISE W3-band flux measurement."""

    name: ClassVar[str] = "FLUX_W3"
    token_key: ClassVar[str] = "tok_flux_w3"


class LegacySurveyFluxW4(Scalar, Modality):
    """WISE W4-band flux measurement."""

    name: ClassVar[str] = "FLUX_W4"
    token_key: ClassVar[str] = "tok_flux_w4"


# Shape parameters
class LegacySurveyShapeR(Scalar, Modality):
    """R-band shape measurement (e.g., half-light radius)."""

    name: ClassVar[str] = "SHAPE_R"
    token_key: ClassVar[str] = "tok_shape_r"


class LegacySurveyShapeE1(Scalar, Modality):
    """First ellipticity component."""

    name: ClassVar[str] = "SHAPE_E1"
    token_key: ClassVar[str] = "tok_shape_e1"


class LegacySurveyShapeE2(Scalar, Modality):
    """Second ellipticity component."""

    name: ClassVar[str] = "SHAPE_E2"
    token_key: ClassVar[str] = "tok_shape_e2"


# Other scalar properties
class LegacySurveyEBV(Scalar, Modality):
    """E(B-V) extinction measurement."""

    name: ClassVar[str] = "EBV"
    token_key: ClassVar[str] = "tok_ebv"


# Spectroscopic redshift
class Z(Scalar, Modality):
    """Spectroscopic redshift measurement."""

    name: ClassVar[str] = "Z"
    token_key: ClassVar[str] = "tok_z"


# Extinction values from HSC
class HSCAG(Scalar, Modality):
    """HSC a_g extinction."""

    name: ClassVar[str] = "a_g"
    token_key: ClassVar[str] = "tok_a_g"


class HSCAR(Scalar, Modality):
    """HSC a_r extinction."""

    name: ClassVar[str] = "a_r"
    token_key: ClassVar[str] = "tok_a_r"


class HSCAI(Scalar, Modality):
    """HSC a_i extinction."""

    name: ClassVar[str] = "a_i"
    token_key: ClassVar[str] = "tok_a_i"


class HSCAZ(Scalar, Modality):
    """HSC a_z extinction."""

    name: ClassVar[str] = "a_z"
    token_key: ClassVar[str] = "tok_a_z"


class HSCAY(Scalar, Modality):
    """HSC a_y extinction."""

    name: ClassVar[str] = "a_y"
    token_key: ClassVar[str] = "tok_a_y"


class HSCMagG(Scalar, Modality):
    """HSC g-band cmodel magnitude."""

    name: ClassVar[str] = "g_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_g"


class HSCMagR(Scalar, Modality):
    """HSC r-band cmodel magnitude."""

    name: ClassVar[str] = "r_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_r"


class HSCMagI(Scalar, Modality):
    """HSC i-band cmodel magnitude."""

    name: ClassVar[str] = "i_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_i"


class HSCMagZ(Scalar, Modality):
    """HSC z-band cmodel magnitude."""

    name: ClassVar[str] = "z_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_z"


class HSCMagY(Scalar, Modality):
    """HSC y-band cmodel magnitude."""

    name: ClassVar[str] = "y_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_y"


class HSCShape11(Scalar, Modality):
    """HSC i-band SDSS shape 11 component."""

    name: ClassVar[str] = "i_sdssshape_shape11"
    token_key: ClassVar[str] = "tok_shape11"


class HSCShape22(Scalar, Modality):
    """HSC i-band SDSS shape 22 component."""

    name: ClassVar[str] = "i_sdssshape_shape22"
    token_key: ClassVar[str] = "tok_shape22"


class HSCShape12(Scalar, Modality):
    """HSC i-band SDSS shape 12 component."""

    name: ClassVar[str] = "i_sdssshape_shape12"
    token_key: ClassVar[str] = "tok_shape12"


# Gaia modalities
class GaiaFluxG(Scalar, Modality):
    """Gaia G-band mean flux."""

    name: ClassVar[str] = "phot_g_mean_flux"
    token_key: ClassVar[str] = "tok_flux_g_gaia"


class GaiaFluxBp(Scalar, Modality):
    """Gaia BP-band mean flux."""

    name: ClassVar[str] = "phot_bp_mean_flux"
    token_key: ClassVar[str] = "tok_flux_bp_gaia"


class GaiaFluxRp(Scalar, Modality):
    """Gaia RP-band mean flux."""

    name: ClassVar[str] = "phot_rp_mean_flux"
    token_key: ClassVar[str] = "tok_flux_rp_gaia"


class GaiaParallax(Scalar, Modality):
    """Gaia parallax measurement."""

    name: ClassVar[str] = "parallax"
    token_key: ClassVar[str] = "tok_parallax"


class Ra(Scalar, Modality):
    """Right ascension coordinate."""

    name: ClassVar[str] = "ra"
    token_key: ClassVar[str] = "tok_ra"


class Dec(Scalar, Modality):
    """Declination coordinate."""

    name: ClassVar[str] = "dec"
    token_key: ClassVar[str] = "tok_dec"


class GaiaXpBp(Scalar, Modality):
    """Gaia BP spectral coefficients."""

    name: ClassVar[str] = "bp_coefficients"
    token_key: ClassVar[str] = "tok_xp_bp"


class GaiaXpRp(Scalar, Modality):
    """Gaia RP spectral coefficients."""

    name: ClassVar[str] = "rp_coefficients"
    token_key: ClassVar[str] = "tok_xp_rp"


ScalarModalities = [
    LegacySurveyFluxG,
    LegacySurveyFluxR,
    LegacySurveyFluxI,
    LegacySurveyFluxZ,
    LegacySurveyFluxW1,
    LegacySurveyFluxW2,
    LegacySurveyFluxW3,
    LegacySurveyFluxW4,
    LegacySurveyShapeR,
    LegacySurveyShapeE1,
    LegacySurveyShapeE2,
    LegacySurveyEBV,
    Z,
    HSCAG,
    HSCAR,
    HSCAI,
    HSCAZ,
    HSCAY,
    HSCMagG,
    HSCMagR,
    HSCMagI,
    HSCMagZ,
    HSCMagY,
    HSCShape11,
    HSCShape22,
    HSCShape12,
    GaiaFluxG,
    GaiaFluxBp,
    GaiaFluxRp,
    GaiaParallax,
    Ra,
    Dec,
    GaiaXpBp,
    GaiaXpRp,
]

# Convenience type for any modality data
ModalityType = (
    Image | Spectrum | Scalar | LegacySurveyCatalog | LegacySurveySegmentationMap
)
