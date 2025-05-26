from dataclasses import dataclass, field, fields
from abc import ABC
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


@dataclass
class Modality(ABC):
    """Base class for all modality data types."""

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}


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
class HSCImage(Image):
    """HSC image modality data."""

    token_key: str = field(init=False, default="tok_image_hsc")


@dataclass
class LegacySurveyImage(Image):
    """Legacy Survey image modality data."""

    token_key: str = field(init=False, default="tok_image")


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
class DESISpectrum(Spectrum):
    """DESI spectrum modality data."""

    token_key: str = field(init=False, default="tok_spectrum_desi")


@dataclass
class SDSSSpectrum(Spectrum):
    """SDSS spectrum modality data."""

    token_key: str = field(init=False, default="tok_spectrum_sdss")


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
    token_key: str = field(init=False, default="catalog")


@dataclass
class LegacySurveySegmentationMap(Modality):
    """Legacy Survey segmentation map modality data.

    Represents 2D segmentation maps built from Legacy Survey detections.
    """

    field: Float[Tensor, " batch height width"]
    token_key: str = field(init=False, default="tok_segmap")

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
class LegacySurveyFluxG(Scalar):
    """G-band flux measurement from Legacy Survey."""

    name: str = field(init=False, default="FLUX_G")
    token_key: str = field(init=False, default="tok_flux_g")


@dataclass
class LegacySurveyFluxR(Scalar):
    """R-band flux measurement."""

    name: str = field(init=False, default="FLUX_R")
    token_key: str = field(init=False, default="tok_flux_r")


@dataclass
class LegacySurveyFluxI(Scalar):
    """I-band flux measurement."""

    name: str = field(init=False, default="FLUX_I")
    token_key: str = field(init=False, default="tok_flux_i")


@dataclass
class LegacySurveyFluxZ(Scalar):
    """Z-band flux measurement."""

    name: str = field(init=False, default="FLUX_Z")
    token_key: str = field(init=False, default="tok_flux_z")


@dataclass
class LegacySurveyFluxW1(Scalar):
    """WISE W1-band flux measurement."""

    name: str = field(init=False, default="FLUX_W1")
    token_key: str = field(init=False, default="tok_flux_w1")


@dataclass
class LegacySurveyFluxW2(Scalar):
    """WISE W2-band flux measurement."""

    name: str = field(init=False, default="FLUX_W2")
    token_key: str = field(init=False, default="tok_flux_w2")


@dataclass
class LegacySurveyFluxW3(Scalar):
    """WISE W3-band flux measurement."""

    name: str = field(init=False, default="FLUX_W3")
    token_key: str = field(init=False, default="tok_flux_w3")


@dataclass
class LegacySurveyFluxW4(Scalar):
    """WISE W4-band flux measurement."""

    name: str = field(init=False, default="FLUX_W4")
    token_key: str = field(init=False, default="tok_flux_w4")


# Shape parameters
@dataclass
class LegacySurveyShapeR(Scalar):
    """R-band shape measurement (e.g., half-light radius)."""

    name: str = field(init=False, default="SHAPE_R")
    token_key: str = field(init=False, default="tok_shape_r")


@dataclass
class LegacySurveyShapeE1(Scalar):
    """First ellipticity component."""

    name: str = field(init=False, default="SHAPE_E1")
    token_key: str = field(init=False, default="tok_shape_e1")


@dataclass
class LegacySurveyShapeE2(Scalar):
    """Second ellipticity component."""

    name: str = field(init=False, default="SHAPE_E2")
    token_key: str = field(init=False, default="tok_shape_e2")


# Other scalar properties
@dataclass
class LegacySurveyEBV(Scalar):
    """E(B-V) extinction measurement."""

    name: str = field(init=False, default="EBV")
    token_key: str = field(init=False, default="tok_ebv")


# Spectroscopic redshift
@dataclass
class Z(Scalar):
    """Spectroscopic redshift measurement."""

    name: str = field(init=False, default="Z")
    token_key: str = field(init=False, default="tok_z")


# Extinction values from HSC
@dataclass
class HSCAG(Scalar):
    """HSC a_g extinction."""

    name: str = field(init=False, default="a_g")
    token_key: str = field(init=False, default="tok_a_g")


@dataclass
class HSCAR(Scalar):
    """HSC a_r extinction."""

    name: str = field(init=False, default="a_r")
    token_key: str = field(init=False, default="tok_a_r")


@dataclass
class HSCAI(Scalar):
    """HSC a_i extinction."""

    name: str = field(init=False, default="a_i")
    token_key: str = field(init=False, default="tok_a_i")


@dataclass
class HSCAZ(Scalar):
    """HSC a_z extinction."""

    name: str = field(init=False, default="a_z")
    token_key: str = field(init=False, default="tok_a_z")


@dataclass
class HSCAY(Scalar):
    """HSC a_y extinction."""

    name: str = field(init=False, default="a_y")
    token_key: str = field(init=False, default="tok_a_y")


@dataclass
class HSCMagG(Scalar):
    """HSC g-band cmodel magnitude."""

    name: str = field(init=False, default="g_cmodel_mag")
    token_key: str = field(init=False, default="tok_mag_g")


@dataclass
class HSCMagR(Scalar):
    """HSC r-band cmodel magnitude."""

    name: str = field(init=False, default="r_cmodel_mag")
    token_key: str = field(init=False, default="tok_mag_r")


@dataclass
class HSCMagI(Scalar):
    """HSC i-band cmodel magnitude."""

    name: str = field(init=False, default="i_cmodel_mag")
    token_key: str = field(init=False, default="tok_mag_i")


@dataclass
class HSCMagZ(Scalar):
    """HSC z-band cmodel magnitude."""

    name: str = field(init=False, default="z_cmodel_mag")
    token_key: str = field(init=False, default="tok_mag_z")


@dataclass
class HSCMagY(Scalar):
    """HSC y-band cmodel magnitude."""

    name: str = field(init=False, default="y_cmodel_mag")
    token_key: str = field(init=False, default="tok_mag_y")


@dataclass
class HSCShape11(Scalar):
    """HSC i-band SDSS shape 11 component."""

    name: str = field(init=False, default="i_sdssshape_shape11")
    token_key: str = field(init=False, default="tok_shape11")


@dataclass
class HSCShape22(Scalar):
    """HSC i-band SDSS shape 22 component."""

    name: str = field(init=False, default="i_sdssshape_shape22")
    token_key: str = field(init=False, default="tok_shape22")


@dataclass
class HSCShape12(Scalar):
    """HSC i-band SDSS shape 12 component."""

    name: str = field(init=False, default="i_sdssshape_shape12")
    token_key: str = field(init=False, default="tok_shape12")


# Gaia modalities
@dataclass
class GaiaFluxG(Scalar):
    """Gaia G-band mean flux."""

    name: str = field(init=False, default="phot_g_mean_flux")
    token_key: str = field(init=False, default="tok_flux_g_gaia")


@dataclass
class GaiaFluxBp(Scalar):
    """Gaia BP-band mean flux."""

    name: str = field(init=False, default="phot_bp_mean_flux")
    token_key: str = field(init=False, default="tok_flux_bp_gaia")


@dataclass
class GaiaFluxRp(Scalar):
    """Gaia RP-band mean flux."""

    name: str = field(init=False, default="phot_rp_mean_flux")
    token_key: str = field(init=False, default="tok_flux_rp_gaia")


@dataclass
class GaiaParallax(Scalar):
    """Gaia parallax measurement."""

    name: str = field(init=False, default="parallax")
    token_key: str = field(init=False, default="tok_parallax")


@dataclass
class Ra(Scalar):
    """Right ascension coordinate."""

    name: str = field(init=False, default="ra")
    token_key: str = field(init=False, default="tok_ra")


@dataclass
class Dec(Scalar):
    """Declination coordinate."""

    name: str = field(init=False, default="dec")
    token_key: str = field(init=False, default="tok_dec")


@dataclass
class GaiaXpBp(Scalar):
    """Gaia BP spectral coefficients."""

    name: str = field(init=False, default="bp_coefficients")
    token_key: str = field(init=False, default="tok_xp_bp")


@dataclass
class GaiaXpRp(Scalar):
    """Gaia RP spectral coefficients."""

    name: str = field(init=False, default="rp_coefficients")
    token_key: str = field(init=False, default="tok_xp_rp")


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
