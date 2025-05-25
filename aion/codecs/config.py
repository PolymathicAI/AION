# Configuration for codecs

from aion.modalities import (
    Image,
    Spectrum,
    LegacySurveyCatalog,
    LegacySurveySegmentationMap,
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
)
from aion.codecs.image import ImageCodec
from aion.codecs.spectrum import SpectrumCodec
from aion.codecs.catalog import CatalogCodec
from aion.codecs.scalar_field import ScalarFieldCodec
from aion.codecs.scalar import (
    ScalarCodec,
    LogScalarCodec,
    MultiScalarCodec,
    GridScalarCodec,
)

CODEC_CONFIG = {
    Image: {
        "class": ImageCodec,
        "repo_id": "polymathic-ai/aion-image-codec",
    },
    Spectrum: {
        "class": SpectrumCodec,
        "repo_id": "polymathic-ai/aion-spectrum-codec",
    },
    LegacySurveyCatalog: {
        "class": CatalogCodec,
        "repo_id": "polymathic-ai/aion-catalog-codec",
    },
    LegacySurveySegmentationMap: {
        "class": ScalarFieldCodec,
        "repo_id": "polymathic-ai/aion-scalar-field-codec",
    },
    # Scalar modalities
    # LogScalarCodec
    LegacySurveyFluxG: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-flux-g-codec",
    },
    LegacySurveyFluxR: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-flux-r-codec",
    },
    LegacySurveyFluxI: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-flux-i-codec",
    },
    LegacySurveyFluxZ: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-flux-z-codec",
    },
    LegacySurveyFluxW1: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-flux-w1-codec",
    },
    LegacySurveyFluxW2: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-flux-w2-codec",
    },
    LegacySurveyFluxW3: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-flux-w3-codec",
    },
    LegacySurveyFluxW4: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-flux-w4-codec",
    },
    LegacySurveyShapeR: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-shape-r-codec",
    },
    GaiaFluxG: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-phot-g-mean-flux-codec",
    },
    GaiaFluxBp: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-phot-bp-mean-flux-codec",
    },
    GaiaFluxRp: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-phot-rp-mean-flux-codec",
    },
    GaiaParallax: {
        "class": LogScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-parallax-codec",
    },
    # ScalarCodec
    LegacySurveyShapeE1: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-shape-e1-codec",
    },
    LegacySurveyShapeE2: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-shape-e2-codec",
    },
    LegacySurveyEBV: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-ebv-codec",
    },
    HSCMagG: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-g-cmodel-mag-codec",
    },
    HSCMagR: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-r-cmodel-mag-codec",
    },
    HSCMagI: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-i-cmodel-mag-codec",
    },
    HSCMagZ: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-z-cmodel-mag-codec",
    },
    HSCMagY: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-y-cmodel-mag-codec",
    },
    HSCShape11: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-i-sdssshape-shape11-codec",
    },
    HSCShape22: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-i-sdssshape-shape22-codec",
    },
    HSCShape12: {
        "class": ScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-i-sdssshape-shape12-codec",
    },
    HSCAG: {"class": ScalarCodec, "repo_id": "polymathic-ai/aion-scalar-a-g-codec"},
    HSCAR: {"class": ScalarCodec, "repo_id": "polymathic-ai/aion-scalar-a-r-codec"},
    HSCAI: {"class": ScalarCodec, "repo_id": "polymathic-ai/aion-scalar-a-i-codec"},
    HSCAZ: {"class": ScalarCodec, "repo_id": "polymathic-ai/aion-scalar-a-z-codec"},
    HSCAY: {"class": ScalarCodec, "repo_id": "polymathic-ai/aion-scalar-a-y-codec"},
    Ra: {"class": ScalarCodec, "repo_id": "polymathic-ai/aion-scalar-ra-codec"},
    Dec: {"class": ScalarCodec, "repo_id": "polymathic-ai/aion-scalar-dec-codec"},
    # MultiScalarCodec
    GaiaXpBp: {
        "class": MultiScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-bp-coefficients-codec",
    },
    GaiaXpRp: {
        "class": MultiScalarCodec,
        "repo_id": "polymathic-ai/aion-scalar-rp-coefficients-codec",
    },
    # GridScalarCodec
    Z: {"class": GridScalarCodec, "repo_id": "polymathic-ai/aion-scalar-z-codec"},
}
