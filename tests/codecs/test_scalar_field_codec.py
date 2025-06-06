import torch

from aion.codecs import ScalarFieldCodec
from aion.modalities import LegacySurveySegmentationMap


def test_scalar_field_tokenizer(data_dir):
    codec = ScalarFieldCodec.from_pretrained("polymathic-ai/aion-scalar-field-codec")
    codec.eval()
    input_batch = torch.load(
        data_dir / "scalar-field_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_batch = torch.load(
        data_dir / "scalar-field_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_batch = torch.load(
        data_dir / "scalar-field_codec_decoded_batch.pt", weights_only=False
    )

    # We flatten the reference encoded output to match the encoded output
    # as we now make all codecs return flattened outputs
    reference_encoded_batch = reference_encoded_batch.reshape(
        reference_encoded_batch.shape[0], -1
    )

    with torch.no_grad():
        output = codec.encode(LegacySurveySegmentationMap(field=input_batch))
        decoded_output = codec.decode(output)

    assert torch.allclose(output, reference_encoded_batch)
    assert torch.allclose(
        decoded_output.field, reference_decoded_batch, atol=1e-4, rtol=1e-4
    )
