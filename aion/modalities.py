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


class Modality(ABC):
    """Base class for all modality data types."""

    token_key: ClassVar[str] = ""


@dataclass
class Image(Modality):
    """Base class for image modality data.

    This is an abstract base class. Use LegacySurveyImage or HSCImage instead.
    """

    name: ClassVar[str] = "image"
    flux: Float[Tensor, " batch num_bands height width"]
    bands: list[str]

    def __repr__(self) -> str:
        repr_str = f"Image(flux_shape={list(self.flux.shape)}, bands={self.bands})"
        return repr_str


class HSCImage(Image):
    """HSC image modality data."""

    token_key: ClassVar[str] = "tok_image_hsc"
    num_tokens: ClassVar[int] = 576


class LegacySurveyImage(Image):
    """Legacy Survey image modality data."""

    token_key: ClassVar[str] = "tok_image"
    num_tokens: ClassVar[int] = 576


@dataclass
class Spectrum(Modality):
    """Base class for spectrum modality data.

    This is an abstract base class. Use DESISpectrum or SDSSSpectrum instead.
    """

    name: ClassVar[str] = "spectrum"
    flux: Float[Tensor, " batch length"]
    ivar: Float[Tensor, " batch length"]
    mask: Bool[Tensor, " batch length"]
    wavelength: Float[Tensor, " batch length"]
    pad_length: ClassVar[int]

    def __repr__(self) -> str:
        repr_str = (
            f"Spectrum(flux_shape={list(self.flux.shape)}, "
            f"wavelength_range=[{self.wavelength.min().item():.1f}, "
            f"{self.wavelength.max().item():.1f}])"
        )
        return repr_str


class DESISpectrum(Spectrum):
    """DESI spectrum modality data."""

    token_key: ClassVar[str] = "tok_spectrum_desi"
    num_tokens: ClassVar[int] = 273
    pad_length: ClassVar[int] = 7808


class SDSSSpectrum(Spectrum):
    """SDSS spectrum modality data."""

    token_key: ClassVar[str] = "tok_spectrum_sdss"
    num_tokens: ClassVar[int] = 273
    pad_length: ClassVar[int] = 4800


# Catalog modality
@dataclass
class LegacySurveyCatalog(Modality):
    """Catalog modality data.

    Represents a catalog of scalar values from the Legacy Survey.
    """

    name: ClassVar[str] = "catalog"
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

    name: ClassVar[str] = "segmentation_map"
    field: Float[Tensor, " batch height width"]
    token_key: ClassVar[str] = "tok_segmap"

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}>(field_shape={list(self.field.shape)})"
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
class LegacySurveyFluxG(Scalar):
    """G-band flux measurement from Legacy Survey."""

    name: ClassVar[str] = "FLUX_G"
    token_key: ClassVar[str] = "tok_flux_g"
    num_tokens: ClassVar[int] = 1


class LegacySurveyFluxR(Scalar):
    """R-band flux measurement."""

    name: ClassVar[str] = "FLUX_R"
    token_key: ClassVar[str] = "tok_flux_r"
    num_tokens: ClassVar[int] = 1


class LegacySurveyFluxI(Scalar):
    """I-band flux measurement."""

    name: ClassVar[str] = "FLUX_I"
    token_key: ClassVar[str] = "tok_flux_i"
    num_tokens: ClassVar[int] = 1


class LegacySurveyFluxZ(Scalar):
    """Z-band flux measurement."""

    name: ClassVar[str] = "FLUX_Z"
    token_key: ClassVar[str] = "tok_flux_z"
    num_tokens: ClassVar[int] = 1


class LegacySurveyFluxW1(Scalar):
    """WISE W1-band flux measurement."""

    name: ClassVar[str] = "FLUX_W1"
    token_key: ClassVar[str] = "tok_flux_w1"
    num_tokens: ClassVar[int] = 1


class LegacySurveyFluxW2(Scalar):
    """WISE W2-band flux measurement."""

    name: ClassVar[str] = "FLUX_W2"
    token_key: ClassVar[str] = "tok_flux_w2"
    num_tokens: ClassVar[int] = 1


class LegacySurveyFluxW3(Scalar):
    """WISE W3-band flux measurement."""

    name: ClassVar[str] = "FLUX_W3"
    token_key: ClassVar[str] = "tok_flux_w3"
    num_tokens: ClassVar[int] = 1


class LegacySurveyFluxW4(Scalar):
    """WISE W4-band flux measurement."""

    name: ClassVar[str] = "FLUX_W4"
    token_key: ClassVar[str] = "tok_flux_w4"
    num_tokens: ClassVar[int] = 1


# Shape parameters
class LegacySurveyShapeR(Scalar):
    """R-band shape measurement (e.g., half-light radius)."""

    name: ClassVar[str] = "SHAPE_R"
    token_key: ClassVar[str] = "tok_shape_r"
    num_tokens: ClassVar[int] = 1


class LegacySurveyShapeE1(Scalar):
    """First ellipticity component."""

    name: ClassVar[str] = "SHAPE_E1"
    token_key: ClassVar[str] = "tok_shape_e1"
    num_tokens: ClassVar[int] = 1


class LegacySurveyShapeE2(Scalar):
    """Second ellipticity component."""

    name: ClassVar[str] = "SHAPE_E2"
    token_key: ClassVar[str] = "tok_shape_e2"
    num_tokens: ClassVar[int] = 1


# Other scalar properties
class LegacySurveyEBV(Scalar):
    """E(B-V) extinction measurement."""

    name: ClassVar[str] = "EBV"
    token_key: ClassVar[str] = "tok_ebv"
    num_tokens: ClassVar[int] = 1


# Spectroscopic redshift
class Z(Scalar):
    """Spectroscopic redshift measurement."""

    name: ClassVar[str] = "Z"
    token_key: ClassVar[str] = "tok_z"
    num_tokens: ClassVar[int] = 1


# Extinction values from HSC
class HSCAG(Scalar):
    """HSC a_g extinction."""

    name: ClassVar[str] = "a_g"
    token_key: ClassVar[str] = "tok_a_g"
    num_tokens: ClassVar[int] = 1


class HSCAR(Scalar):
    """HSC a_r extinction."""

    name: ClassVar[str] = "a_r"
    token_key: ClassVar[str] = "tok_a_r"
    num_tokens: ClassVar[int] = 1


class HSCAI(Scalar):
    """HSC a_i extinction."""

    name: ClassVar[str] = "a_i"
    token_key: ClassVar[str] = "tok_a_i"
    num_tokens: ClassVar[int] = 1


class HSCAZ(Scalar):
    """HSC a_z extinction."""

    name: ClassVar[str] = "a_z"
    token_key: ClassVar[str] = "tok_a_z"
    num_tokens: ClassVar[int] = 1


class HSCAY(Scalar):
    """HSC a_y extinction."""

    name: ClassVar[str] = "a_y"
    token_key: ClassVar[str] = "tok_a_y"
    num_tokens: ClassVar[int] = 1


class HSCMagG(Scalar):
    """HSC g-band cmodel magnitude."""

    name: ClassVar[str] = "g_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_g"
    num_tokens: ClassVar[int] = 1


class HSCMagR(Scalar):
    """HSC r-band cmodel magnitude."""

    name: ClassVar[str] = "r_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_r"
    num_tokens: ClassVar[int] = 1


class HSCMagI(Scalar):
    """HSC i-band cmodel magnitude."""

    name: ClassVar[str] = "i_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_i"
    num_tokens: ClassVar[int] = 1


class HSCMagZ(Scalar):
    """HSC z-band cmodel magnitude."""

    name: ClassVar[str] = "z_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_z"
    num_tokens: ClassVar[int] = 1


class HSCMagY(Scalar):
    """HSC y-band cmodel magnitude."""

    name: ClassVar[str] = "y_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_y"
    num_tokens: ClassVar[int] = 1


class HSCShape11(Scalar):
    """HSC i-band SDSS shape 11 component."""

    name: ClassVar[str] = "i_sdssshape_shape11"
    token_key: ClassVar[str] = "tok_shape11"
    num_tokens: ClassVar[int] = 1


class HSCShape22(Scalar):
    """HSC i-band SDSS shape 22 component."""

    name: ClassVar[str] = "i_sdssshape_shape22"
    token_key: ClassVar[str] = "tok_shape22"
    num_tokens: ClassVar[int] = 1


class HSCShape12(Scalar):
    """HSC i-band SDSS shape 12 component."""

    name: ClassVar[str] = "i_sdssshape_shape12"
    token_key: ClassVar[str] = "tok_shape12"
    num_tokens: ClassVar[int] = 1


# Gaia modalities
class GaiaFluxG(Scalar):
    """Gaia G-band mean flux."""

    name: ClassVar[str] = "phot_g_mean_flux"
    token_key: ClassVar[str] = "tok_flux_g_gaia"
    num_tokens: ClassVar[int] = 1


class GaiaFluxBp(Scalar):
    """Gaia BP-band mean flux."""

    name: ClassVar[str] = "phot_bp_mean_flux"
    token_key: ClassVar[str] = "tok_flux_bp_gaia"
    num_tokens: ClassVar[int] = 1


class GaiaFluxRp(Scalar):
    """Gaia RP-band mean flux."""

    name: ClassVar[str] = "phot_rp_mean_flux"
    token_key: ClassVar[str] = "tok_flux_rp_gaia"
    num_tokens: ClassVar[int] = 1


class GaiaParallax(Scalar):
    """Gaia parallax measurement."""

    name: ClassVar[str] = "parallax"
    token_key: ClassVar[str] = "tok_parallax"
    num_tokens: ClassVar[int] = 1


class Ra(Scalar):
    """Right ascension coordinate."""

    name: ClassVar[str] = "ra"
    token_key: ClassVar[str] = "tok_ra"
    num_tokens: ClassVar[int] = 1


class Dec(Scalar):
    """Declination coordinate."""

    name: ClassVar[str] = "dec"
    token_key: ClassVar[str] = "tok_dec"
    num_tokens: ClassVar[int] = 1


class GaiaXpBp(Scalar):
    """Gaia BP spectral coefficients."""

    name: ClassVar[str] = "bp_coefficients"
    token_key: ClassVar[str] = "tok_xp_bp"
    num_tokens: ClassVar[int] = 55


class GaiaXpRp(Scalar):
    """Gaia RP spectral coefficients."""

    name: ClassVar[str] = "rp_coefficients"
    token_key: ClassVar[str] = "tok_xp_rp"
    num_tokens: ClassVar[int] = 55


ScalarModalities = {
    modality.name: modality
    for modality in [
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
}

# Convenience type for any modality data
ModalityType = (
    Image | Spectrum | Scalar | LegacySurveyCatalog | LegacySurveySegmentationMap
)
