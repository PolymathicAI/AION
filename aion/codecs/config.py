from dataclasses import dataclass
from typing import TypeVar

from aion.codecs.catalog import CatalogCodec
from aion.codecs.image import ImageCodec
from aion.codecs.scalar import (
    GridScalarCodec,
    LogScalarCodec,
    MultiScalarCodec,
    ScalarCodec,
)
from aion.codecs.scalar_field import ScalarFieldCodec
from aion.codecs.spectrum import SpectrumCodec
from aion.modalities import (
    HSCAG,
    HSCAI,
    HSCAR,
    HSCAY,
    HSCAZ,
    Dec,
    GaiaFluxBp,
    GaiaFluxG,
    GaiaFluxRp,
    GaiaParallax,
    GaiaXpBp,
    GaiaXpRp,
    HSCMagG,
    HSCMagI,
    HSCMagR,
    HSCMagY,
    HSCMagZ,
    HSCShape11,
    HSCShape12,
    HSCShape22,
    Image,
    LegacySurveyCatalog,
    LegacySurveyEBV,
    LegacySurveyFluxG,
    LegacySurveyFluxI,
    LegacySurveyFluxR,
    LegacySurveyFluxW1,
    LegacySurveyFluxW2,
    LegacySurveyFluxW3,
    LegacySurveyFluxW4,
    LegacySurveyFluxZ,
    LegacySurveySegmentationMap,
    LegacySurveyShapeE1,
    LegacySurveyShapeE2,
    LegacySurveyShapeR,
    Ra,
    Spectrum,
    Z,
)

CodecType = TypeVar(
    "CodecModel",
    bound=type[
        CatalogCodec
        | GridScalarCodec
        | ImageCodec
        | LogScalarCodec
        | MultiScalarCodec
        | ScalarCodec
        | ScalarFieldCodec
        | SpectrumCodec
    ],
)


@dataclass
class CodecHFConfig:
    """Codec configuration for AION."""

    codec_class: CodecType
    repo_id: str


CODEC_CONFIG = {
    Image: CodecHFConfig(
        codec_class=ImageCodec, repo_id="polymathic-ai/aion-image-codec"
    ),
    Spectrum: CodecHFConfig(
        codec_class=SpectrumCodec, repo_id="polymathic-ai/aion-spectrum-codec"
    ),
    LegacySurveyCatalog: CodecHFConfig(
        codec_class=CatalogCodec, repo_id="polymathic-ai/aion-catalog-codec"
    ),
    LegacySurveySegmentationMap: CodecHFConfig(
        codec_class=ScalarFieldCodec, repo_id="polymathic-ai/aion-scalar-field-codec"
    ),
    # Scalar modalities
    # LogScalarCodec
    LegacySurveyFluxG: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-flux-g-codec"
    ),
    LegacySurveyFluxR: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-flux-r-codec"
    ),
    LegacySurveyFluxI: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-flux-i-codec"
    ),
    LegacySurveyFluxZ: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-flux-z-codec"
    ),
    LegacySurveyFluxW1: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-flux-w1-codec"
    ),
    LegacySurveyFluxW2: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-flux-w2-codec"
    ),
    LegacySurveyFluxW3: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-flux-w3-codec"
    ),
    LegacySurveyFluxW4: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-flux-w4-codec"
    ),
    LegacySurveyShapeR: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-shape-r-codec"
    ),
    GaiaFluxG: CodecHFConfig(
        codec_class=LogScalarCodec,
        repo_id="polymathic-ai/aion-scalar-phot-g-mean-flux-codec",
    ),
    GaiaFluxBp: CodecHFConfig(
        codec_class=LogScalarCodec,
        repo_id="polymathic-ai/aion-scalar-phot-bp-mean-flux-codec",
    ),
    GaiaFluxRp: CodecHFConfig(
        codec_class=LogScalarCodec,
        repo_id="polymathic-ai/aion-scalar-phot-rp-mean-flux-codec",
    ),
    GaiaParallax: CodecHFConfig(
        codec_class=LogScalarCodec, repo_id="polymathic-ai/aion-scalar-parallax-codec"
    ),
    # ScalarCodec
    LegacySurveyShapeE1: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-shape-e1-codec"
    ),
    LegacySurveyShapeE2: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-shape-e2-codec"
    ),
    LegacySurveyEBV: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-ebv-codec"
    ),
    HSCMagG: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-g-cmodel-mag-codec"
    ),
    HSCMagR: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-r-cmodel-mag-codec"
    ),
    HSCMagI: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-i-cmodel-mag-codec"
    ),
    HSCMagZ: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-z-cmodel-mag-codec"
    ),
    HSCMagY: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-y-cmodel-mag-codec"
    ),
    HSCShape11: CodecHFConfig(
        codec_class=ScalarCodec,
        repo_id="polymathic-ai/aion-scalar-i-sdssshape-shape11-codec",
    ),
    HSCShape22: CodecHFConfig(
        codec_class=ScalarCodec,
        repo_id="polymathic-ai/aion-scalar-i-sdssshape-shape22-codec",
    ),
    HSCShape12: CodecHFConfig(
        codec_class=ScalarCodec,
        repo_id="polymathic-ai/aion-scalar-i-sdssshape-shape12-codec",
    ),
    HSCAG: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-a-g-codec"
    ),
    HSCAR: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-a-r-codec"
    ),
    HSCAI: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-a-i-codec"
    ),
    HSCAZ: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-a-z-codec"
    ),
    HSCAY: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-a-y-codec"
    ),
    Ra: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-ra-codec"
    ),
    Dec: CodecHFConfig(
        codec_class=ScalarCodec, repo_id="polymathic-ai/aion-scalar-dec-codec"
    ),
    # MultiScalarCodec
    GaiaXpBp: CodecHFConfig(
        codec_class=MultiScalarCodec,
        repo_id="polymathic-ai/aion-scalar-bp-coefficients-codec",
    ),
    GaiaXpRp: CodecHFConfig(
        codec_class=MultiScalarCodec,
        repo_id="polymathic-ai/aion-scalar-rp-coefficients-codec",
    ),
    # GridScalarCodec
    Z: CodecHFConfig(
        codec_class=GridScalarCodec, repo_id="polymathic-ai/aion-scalar-z-codec"
    ),
}
