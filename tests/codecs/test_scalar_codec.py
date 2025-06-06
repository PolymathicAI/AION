import pytest
import torch

from aion.codecs import LogScalarCodec, ScalarCodec, MultiScalarCodec, GridScalarCodec
from aion.modalities import (
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
    HSCMagG,
    HSCMagR,
    HSCMagI,
    HSCMagZ,
    HSCMagY,
    HSCShape11,
    HSCShape22,
    HSCShape12,
    HSCAG,
    HSCAR,
    HSCAI,
    HSCAZ,
    HSCAY,
    # Gaia modalities
    GaiaFluxG,
    GaiaFluxBp,
    GaiaFluxRp,
    GaiaParallax,
    Ra,
    Dec,
    GaiaXpBp,
    GaiaXpRp,
)


@pytest.mark.parametrize(
    "codec_class,modality",
    [
        # LogScalarCodec tests
        (LogScalarCodec, LegacySurveyFluxG),
        (LogScalarCodec, LegacySurveyFluxR),
        (LogScalarCodec, LegacySurveyFluxI),
        (LogScalarCodec, LegacySurveyFluxZ),
        (LogScalarCodec, LegacySurveyFluxW1),
        (LogScalarCodec, LegacySurveyFluxW2),
        (LogScalarCodec, LegacySurveyFluxW3),
        (LogScalarCodec, LegacySurveyFluxW4),
        (LogScalarCodec, LegacySurveyShapeR),
        # Gaia LogScalarCodec tests
        (LogScalarCodec, GaiaFluxG),
        (LogScalarCodec, GaiaFluxBp),
        (LogScalarCodec, GaiaFluxRp),
        (LogScalarCodec, GaiaParallax),
        # ScalarCodec tests
        (ScalarCodec, LegacySurveyShapeE1),
        (ScalarCodec, LegacySurveyShapeE2),
        (ScalarCodec, LegacySurveyEBV),
        (ScalarCodec, HSCMagG),
        (ScalarCodec, HSCMagR),
        (ScalarCodec, HSCMagI),
        (ScalarCodec, HSCMagZ),
        (ScalarCodec, HSCMagY),
        (ScalarCodec, HSCShape11),
        (ScalarCodec, HSCShape22),
        (ScalarCodec, HSCShape12),
        (ScalarCodec, HSCAG),
        (ScalarCodec, HSCAR),
        (ScalarCodec, HSCAI),
        (ScalarCodec, HSCAZ),
        (ScalarCodec, HSCAY),
        # Gaia ScalarCodec tests
        (ScalarCodec, Ra),
        (ScalarCodec, Dec),
        # Gaia MultiScalarCodec tests
        (MultiScalarCodec, GaiaXpBp),
        (MultiScalarCodec, GaiaXpRp),
        # Grid tokenizer
        (GridScalarCodec, Z),
    ],
)
def test_scalar_tokenizer(data_dir, codec_class, modality):
    codec = codec_class.from_pretrained(
        f"polymathic-ai/aion-scalar-{modality.name.lower().replace('_', '-')}-codec"
    )
    codec.eval()
    input_batch = torch.load(
        data_dir / f"{modality.name}_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_batch = torch.load(
        data_dir / f"{modality.name}_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_batch = torch.load(
        data_dir / f"{modality.name}_codec_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        output = codec.encode(modality(value=input_batch))
        decoded_output = codec.decode(output)

    assert torch.allclose(output, reference_encoded_batch)
    assert torch.allclose(decoded_output.value, reference_decoded_batch)
