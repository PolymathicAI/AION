"""Structured modality data types using Pydantic for type safety and validation."""

from typing import List, Union, ClassVar
from pydantic import BaseModel, Field, ConfigDict
from jaxtyping import Float, Bool, Int
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


class Modality(BaseModel):
    """Base class for all modality data types."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class Image(Modality):
    """Base class for image modality data.

    This is an abstract base class. Use LegacySurveyImage or HSCImage instead.
    """

    flux: Float[Tensor, "batch num_bands height width"] = Field(
        description="Array of flux measurements of the image."
    )
    bands: List[str] = Field(
        description="List of band names, indicating the wavelength range of each channel in flux (e.g., ['DES-G', 'DES-R']). Its length must match num_bands."
    )

    def __repr__(self) -> str:
        repr_str = f"Image(flux_shape={list(self.flux.shape)}, bands={self.bands})"
        return repr_str


class HSCImage(Image):
    """HSC image modality data."""

    token_key: ClassVar[str] = "tok_image_hsc"


class LegacySurveyImage(Image):
    """Legacy Survey image modality data."""

    token_key: ClassVar[str] = "tok_image"


class Spectrum(Modality):
    """Base class for spectrum modality data.

    This is an abstract base class. Use DESISpectrum or SDSSSpectrum instead.
    """

    flux: Float[Tensor, "batch length"] = Field(
        description="Array of flux measurements of the spectrum."
    )
    ivar: Float[Tensor, "batch length"] = Field(
        description="Array of inverse variance values for the spectrum."
    )
    mask: Bool[Tensor, "batch length"] = Field(
        description="Mask array indicating valid/invalid values in the spectrum."
    )
    wavelength: Float[Tensor, "batch length"] = Field(
        description="Array of wavelength values in Angstroms."
    )

    def __repr__(self) -> str:
        repr_str = f"Spectrum(flux_shape={list(self.flux.shape)}, wavelength_range=[{self.wavelength.min().item():.1f}, {self.wavelength.max().item():.1f}])"
        return repr_str


class DESISpectrum(Spectrum):
    """DESI spectrum modality data."""

    token_key: ClassVar[str] = "tok_spectrum_desi"


class SDSSSpectrum(Spectrum):
    """SDSS spectrum modality data."""

    token_key: ClassVar[str] = "tok_spectrum_sdss"


# Catalog modality
class LegacySurveyCatalog(Modality):
    """Catalog modality data.

    Represents a catalog of scalar values from the
    Legacy Survey.
    """

    token_key: ClassVar[str] = "catalog"

    X: Int[Tensor, "batch n"] = Field(
        description="X position of the object in the image."
    )
    Y: Int[Tensor, "batch n"] = Field(
        description="Y position of the object in the image."
    )
    SHAPE_E1: Float[Tensor, "batch n"] = Field(
        description="First ellipticity component of the object."
    )
    SHAPE_E2: Float[Tensor, "batch n"] = Field(
        description="Second ellipticity component of the object."
    )
    SHAPE_R: Float[Tensor, "batch n"] = Field(description="Size of the object.")


class LegacySurveySegmentationMap(Modality):
    """Legacy Survey segmentation map modality data.

    Represents 2D segmentation maps built from Legacy Survey detections.
    """

    token_key: ClassVar[str] = "tok_segmap"

    field: Float[Tensor, "batch height width"] = Field(
        description="Segmentation map data with spatial dimensions."
    )

    def __repr__(self) -> str:
        repr_str = f"LegacySurveySegmentationMap(field_shape={list(self.field.shape)})"
        return repr_str


class Scalar(Modality):
    """Base class for scalar modality data.

    Represents a single scalar value per sample, typically used for
    flux measurements, shape parameters, or other single-valued properties.
    """

    name: ClassVar[str] = ""
    value: Float[Tensor, "..."] = Field(
        description="Scalar value for each sample in the batch."
    )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={list(self.value.shape)})"


# Flux measurements in different bands
class LegacySurveyFluxG(Scalar):
    """G-band flux measurement from Legacy Survey."""

    name: ClassVar[str] = "FLUX_G"
    token_key: ClassVar[str] = "tok_flux_g"


class LegacySurveyFluxR(Scalar):
    """R-band flux measurement."""

    name: ClassVar[str] = "FLUX_R"
    token_key: ClassVar[str] = "tok_flux_r"


class LegacySurveyFluxI(Scalar):
    """I-band flux measurement."""

    name: ClassVar[str] = "FLUX_I"
    token_key: ClassVar[str] = "tok_flux_i"


class LegacySurveyFluxZ(Scalar):
    """Z-band flux measurement."""

    name: ClassVar[str] = "FLUX_Z"
    token_key: ClassVar[str] = "tok_flux_z"


class LegacySurveyFluxW1(Scalar):
    """WISE W1-band flux measurement."""

    name: ClassVar[str] = "FLUX_W1"
    token_key: ClassVar[str] = "tok_flux_w1"


class LegacySurveyFluxW2(Scalar):
    """WISE W2-band flux measurement."""

    name: ClassVar[str] = "FLUX_W2"
    token_key: ClassVar[str] = "tok_flux_w2"


class LegacySurveyFluxW3(Scalar):
    """WISE W3-band flux measurement."""

    name: ClassVar[str] = "FLUX_W3"
    token_key: ClassVar[str] = "tok_flux_w3"


class LegacySurveyFluxW4(Scalar):
    """WISE W4-band flux measurement."""

    name: ClassVar[str] = "FLUX_W4"
    token_key: ClassVar[str] = "tok_flux_w4"


# Shape parameters
class LegacySurveyShapeR(Scalar):
    """R-band shape measurement (e.g., half-light radius)."""

    name: ClassVar[str] = "SHAPE_R"
    token_key: ClassVar[str] = "tok_shape_r"


class LegacySurveyShapeE1(Scalar):
    """First ellipticity component."""

    name: ClassVar[str] = "SHAPE_E1"
    token_key: ClassVar[str] = "tok_shape_e1"


class LegacySurveyShapeE2(Scalar):
    """Second ellipticity component."""

    name: ClassVar[str] = "SHAPE_E2"
    token_key: ClassVar[str] = "tok_shape_e2"


# Other scalar properties
class LegacySurveyEBV(Scalar):
    """E(B-V) extinction measurement."""

    name: ClassVar[str] = "EBV"
    token_key: ClassVar[str] = "tok_ebv"


# Spectroscopic redshift
class Z(Scalar):
    """Spectroscopic redshift measurement."""

    name: ClassVar[str] = "Z"
    token_key: ClassVar[str] = "tok_z"


# Extinction values from HSC
class HSCAG(Scalar):
    """HSC a_g extinction."""

    name: ClassVar[str] = "a_g"
    token_key: ClassVar[str] = "tok_a_g"


class HSCAR(Scalar):
    """HSC a_r extinction."""

    name: ClassVar[str] = "a_r"
    token_key: ClassVar[str] = "tok_a_r"


class HSCAI(Scalar):
    """HSC a_i extinction."""

    name: ClassVar[str] = "a_i"
    token_key: ClassVar[str] = "tok_a_i"


class HSCAZ(Scalar):
    """HSC a_z extinction."""

    name: ClassVar[str] = "a_z"
    token_key: ClassVar[str] = "tok_a_z"


class HSCAY(Scalar):
    """HSC a_y extinction."""

    name: ClassVar[str] = "a_y"
    token_key: ClassVar[str] = "tok_a_y"


class HSCMagG(Scalar):
    """HSC g-band cmodel magnitude."""

    name: ClassVar[str] = "g_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_g"


class HSCMagR(Scalar):
    """HSC r-band cmodel magnitude."""

    name: ClassVar[str] = "r_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_r"


class HSCMagI(Scalar):
    """HSC i-band cmodel magnitude."""

    name: ClassVar[str] = "i_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_i"


class HSCMagZ(Scalar):
    """HSC z-band cmodel magnitude."""

    name: ClassVar[str] = "z_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_z"


class HSCMagY(Scalar):
    """HSC y-band cmodel magnitude."""

    name: ClassVar[str] = "y_cmodel_mag"
    token_key: ClassVar[str] = "tok_mag_y"


class HSCShape11(Scalar):
    """HSC i-band SDSS shape 11 component."""

    name: ClassVar[str] = "i_sdssshape_shape11"
    token_key: ClassVar[str] = "tok_shape11"


class HSCShape22(Scalar):
    """HSC i-band SDSS shape 22 component."""

    name: ClassVar[str] = "i_sdssshape_shape22"
    token_key: ClassVar[str] = "tok_shape22"


class HSCShape12(Scalar):
    """HSC i-band SDSS shape 12 component."""

    name: ClassVar[str] = "i_sdssshape_shape12"
    token_key: ClassVar[str] = "tok_shape12"


# Gaia modalities
class GaiaFluxG(Scalar):
    """Gaia G-band mean flux."""

    name: ClassVar[str] = "phot_g_mean_flux"
    token_key: ClassVar[str] = "tok_flux_g_gaia"


class GaiaFluxBp(Scalar):
    """Gaia BP-band mean flux."""

    name: ClassVar[str] = "phot_bp_mean_flux"
    token_key: ClassVar[str] = "tok_flux_bp_gaia"


class GaiaFluxRp(Scalar):
    """Gaia RP-band mean flux."""

    name: ClassVar[str] = "phot_rp_mean_flux"
    token_key: ClassVar[str] = "tok_flux_rp_gaia"


class GaiaParallax(Scalar):
    """Gaia parallax measurement."""

    name: ClassVar[str] = "parallax"
    token_key: ClassVar[str] = "tok_parallax"


class Ra(Scalar):
    """Right ascension coordinate."""

    name: ClassVar[str] = "ra"
    token_key: ClassVar[str] = "tok_ra"


class Dec(Scalar):
    """Declination coordinate."""

    name: ClassVar[str] = "dec"
    token_key: ClassVar[str] = "tok_dec"


class GaiaXpBp(Scalar):
    """Gaia BP spectral coefficients."""

    name: ClassVar[str] = "bp_coefficients"
    token_key: ClassVar[str] = "tok_xp_bp"


class GaiaXpRp(Scalar):
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
ModalityType = Union[
    Image, Spectrum, Scalar, LegacySurveyCatalog, LegacySurveySegmentationMap
]
