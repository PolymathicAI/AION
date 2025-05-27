from dataclasses import dataclass, fields
from abc import ABC
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import ClassVar

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


@dataclass
class Modality(ABC):
    """Base class for all modality data types."""

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}


@dataclass
class TokenModality(Modality):
    """Base class for all token modalities."""

    token_key: ClassVar[str] = ""


@dataclass
class Image(Modality):
    """Base class for image modality data.

    This is an abstract base class. Use LegacySurveyImage or HSCImage instead.
    """

    flux: Float[Tensor, " batch num_bands height width"]
    bands: list[str]

    def __repr__(self) -> str:
        repr_str = f"Image(flux_shape={list(self.flux.shape)}, bands={self.bands})"
        return repr_str


@dataclass
class HSCImage(Image, TokenModality):
    """HSC image modality data."""

    token_key: ClassVar[str] = "tok_image_hsc"


@dataclass
class LegacySurveyImage(Image, TokenModality):
    """Legacy Survey image modality data."""

    token_key: ClassVar[str] = "tok_image"


@dataclass
class Spectrum(Modality):
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


@dataclass
class DESISpectrum(Spectrum, TokenModality):
    """DESI spectrum modality data."""

    token_key: ClassVar[str] = "tok_spectrum_desi"


@dataclass
class SDSSSpectrum(Spectrum, TokenModality):
    """SDSS spectrum modality data."""

    token_key: ClassVar[str] = "tok_spectrum_sdss"


# Catalog modality
@dataclass
class LegacySurveyCatalog(TokenModality):
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
class LegacySurveySegmentationMap(TokenModality):
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
@dataclass
class LegacySurveyFluxG(Scalar, TokenModality):
    """G-band flux measurement from Legacy Survey."""

    name: ClassVar[str] = "FLUX_G"
    token_key: ClassVar[str] = "tok_flux_g"


@dataclass
class LegacySurveyFluxR(Scalar, TokenModality):
    """R-band flux measurement."""

    name: ClassVar[str] = "FLUX_R"
    token_key: ClassVar[str] = "tok_flux_r"


@dataclass
class LegacySurveyFluxI(Scalar, TokenModality):
    """I-band flux measurement."""

    name: ClassVar[str] = "FLUX_I"
    token_key: ClassVar[str] = "tok_flux_i"


@dataclass
class LegacySurveyFluxZ(Scalar, TokenModality):
    """Z-band flux measurement."""

    name: ClassVar[str] = "FLUX_Z"
    token_key: ClassVar[str] = "tok_flux_z"


@dataclass
class LegacySurveyFluxW1(Scalar, TokenModality):
    """WISE W1-band flux measurement."""

    name: ClassVar[str] = "FLUX_W1"
    token_key: ClassVar[str] = "tok_flux_w1"


@dataclass
class LegacySurveyFluxW2(Scalar, TokenModality):
    """WISE W2-band flux measurement."""

    name: ClassVar[str] = "FLUX_W2"
    token_key: ClassVar[str] = "tok_flux_w2"


@dataclass
class LegacySurveyFluxW3(Scalar, TokenModality):
    """WISE W3-band flux measurement."""

    name: ClassVar[str] = "FLUX_W3"
    token_key: ClassVar[str] = "tok_flux_w3"


@dataclass
class LegacySurveyFluxW4(Scalar, TokenModality):
    """WISE W4-band flux measurement."""

    name: ClassVar[str] = "FLUX_W4"
    token_key: ClassVar[str] = "tok_flux_w4"


# Shape parameters
@dataclass
class LegacySurveyShapeR(Scalar, TokenModality):
    """R-band shape measurement (e.g., half-light radius)."""

    name: ClassVar[str] = "SHAPE_R"
    token_key: ClassVar[str] = "tok_shape_r"


@dataclass
class LegacySurveyShapeE1(Scalar, TokenModality):
    """First ellipticity component."""

    name: ClassVar[str] = "SHAPE_E1"
    token_key: ClassVar[str] = "tok_shape_e1"


@dataclass
class LegacySurveyShapeE2(Scalar, TokenModality):
    """Second ellipticity component."""

    name: ClassVar[str] = "SHAPE_E2"
    token_key: ClassVar[str] = "tok_shape_e2"


# Other scalar properties
@dataclass
class LegacySurveyEBV(Scalar, TokenModality):
    """E(B-V) extinction measurement."""

    name: ClassVar[str] = "EBV"
    token_key: ClassVar[str] = "tok_ebv"


# Spectroscopic redshift
@dataclass
class Z(Scalar, TokenModality):
    """Spectroscopic redshift measurement."""

    name: ClassVar[str] = "Z"
    token_key: ClassVar[str] = "tok_z"


# Extinction values from HSC
@dataclass
class HSCAG(Scalar, TokenModality):
    """HSC a_g extinction."""

    name: ClassVar[str] = "a_g"
    token_key: ClassVar[str] = "tok_a_g"


@dataclass
class HSCAR(Scalar, TokenModality):
    """HSC a_r extinction."""

    name: ClassVar[str] = "a_r"
    token_key: ClassVar[str] = "tok_a_r"


@dataclass
class HSCAI(Scalar, TokenModality):
    """HSC a_i extinction."""

    name: ClassVar[str] = "a_i"
    token_key: ClassVar[str] = "tok_a_i"


@dataclass
class HSCAZ(Scalar, TokenModality):
    """HSC a_z extinction."""

    name: ClassVar[str] = "a_z"
    token_key: ClassVar[str] = "tok_a_z"


@dataclass
class HSCAY(Scalar, TokenModality):
    """HSC a_y extinction."""

    name: ClassVar[str] = "a_y"
    token_key: ClassVar[str] = "tok_a_y"


@dataclass
class HSCMagG(Scalar, TokenModality):
    """HSC g-band cmodel magnitude."""

    name: ClassVar[str] = "g_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_g"


@dataclass
class HSCMagR(Scalar, TokenModality):
    """HSC r-band cmodel magnitude."""

    name: ClassVar[str] = "r_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_r"


@dataclass
class HSCMagI(Scalar, TokenModality):
    """HSC i-band cmodel magnitude."""

    name: ClassVar[str] = "i_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_i"


@dataclass
class HSCMagZ(Scalar, TokenModality):
    """HSC z-band cmodel magnitude."""

    name: ClassVar[str] = "z_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_z"


@dataclass
class HSCMagY(Scalar, TokenModality):
    """HSC y-band cmodel magnitude."""

    name: ClassVar[str] = "y_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_y"


@dataclass
class HSCShape11(Scalar, TokenModality):
    """HSC i-band SDSS shape 11 component."""

    name: ClassVar[str] = "i_sdssshape_shape11"
    token_key: ClassVar[str] = "tok_shape11"


@dataclass
class HSCShape22(Scalar, TokenModality):
    """HSC i-band SDSS shape 22 component."""

    name: ClassVar[str] = "i_sdssshape_shape22"
    token_key: ClassVar[str] = "tok_shape22"


@dataclass
class HSCShape12(Scalar, TokenModality):
    """HSC i-band SDSS shape 12 component."""

    name: ClassVar[str] = "i_sdssshape_shape12"
    token_key: ClassVar[str] = "tok_shape12"


# Gaia modalities
@dataclass
class GaiaFluxG(Scalar, TokenModality):
    """Gaia G-band mean flux."""

    name: ClassVar[str] = "phot_g_mean_flux"
    token_key: ClassVar[str] = "tok_flux_g_gaia"


@dataclass
class GaiaFluxBp(Scalar, TokenModality):
    """Gaia BP-band mean flux."""

    name: ClassVar[str] = "phot_bp_mean_flux"
    token_key: ClassVar[str] = "tok_flux_bp_gaia"


@dataclass
class GaiaFluxRp(Scalar, TokenModality):
    """Gaia RP-band mean flux."""

    name: ClassVar[str] = "phot_rp_mean_flux"
    token_key: ClassVar[str] = "tok_flux_rp_gaia"


@dataclass
class GaiaParallax(Scalar, TokenModality):
    """Gaia parallax measurement."""

    name: ClassVar[str] = "parallax"
    token_key: ClassVar[str] = "tok_parallax"


@dataclass
class Ra(Scalar, TokenModality):
    """Right ascension coordinate."""

    name: ClassVar[str] = "ra"
    token_key: ClassVar[str] = "tok_ra"


@dataclass
class Dec(Scalar, TokenModality):
    """Declination coordinate."""

    name: ClassVar[str] = "dec"
    token_key: ClassVar[str] = "tok_dec"


@dataclass
class GaiaXpBp(Scalar, TokenModality):
    """Gaia BP spectral coefficients."""

    name: ClassVar[str] = "bp_coefficients"
    token_key: ClassVar[str] = "tok_xp_bp"


@dataclass
class GaiaXpRp(Scalar, TokenModality):
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
