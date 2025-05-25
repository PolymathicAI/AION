import torch

from aion.codecs import CatalogCodec
from aion.modalities import LegacySurveyCatalog


def test_catalog_tokenizer(data_dir):
    codec = CatalogCodec.from_pretrained("polymathic-ai/aion-catalog-codec")
    codec.eval()
    input_batch = torch.load(
        data_dir / "catalog_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_batch = torch.load(
        data_dir / "catalog_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_batch = torch.load(
        data_dir / "catalog_codec_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        output = codec.encode(LegacySurveyCatalog(**input_batch))
        decoded_output = codec.decode(output)

    assert torch.allclose(output, reference_encoded_batch)
    assert torch.allclose(decoded_output.X, reference_decoded_batch["X"], atol=1e-5)
    assert torch.allclose(decoded_output.Y, reference_decoded_batch["Y"], atol=1e-5)
    assert torch.allclose(
        decoded_output.SHAPE_E1, reference_decoded_batch["SHAPE_E1"], atol=1e-5
    )
    assert torch.allclose(
        decoded_output.SHAPE_E2, reference_decoded_batch["SHAPE_E2"], atol=1e-5
    )
    assert torch.allclose(
        decoded_output.SHAPE_R, reference_decoded_batch["SHAPE_R"], atol=1e-5
    )
