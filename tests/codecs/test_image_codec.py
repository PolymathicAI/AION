import pytest
import torch

from aion.modalities import Image
from aion.codecs import ImageCodec


@pytest.mark.parametrize("embedding_dim", [5, 10])
@pytest.mark.parametrize("multisurvey_projection_dims", [12, 24])
@pytest.mark.parametrize("hidden_dims", [8, 16])
def test_magvit_image_tokenizer(
    embedding_dim, multisurvey_projection_dims, hidden_dims
):
    tokenizer = ImageCodec(
        quantizer_levels=[1] * embedding_dim,
        hidden_dims=hidden_dims,
        multisurvey_projection_dims=multisurvey_projection_dims,
        n_compressions=2,
        num_consecutive=4,
        embedding_dim=embedding_dim,
        range_compression_factor=0.01,
        mult_factor=10,
    )
    batch_size = 4
    flux_tensor = torch.randn(batch_size, 4, 96, 96)
    input_image_obj = Image(
        flux=flux_tensor,
        bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
    )

    encoded = tokenizer.encode(input_image_obj)
    assert encoded.shape == (batch_size, 24 * 24)

    decoded_image_obj = tokenizer.decode(
        encoded, bands=["DES-G", "DES-R", "DES-I", "DES-Z"]
    )

    assert isinstance(decoded_image_obj, Image)
    assert decoded_image_obj.flux.shape == flux_tensor.shape


def test_hf_previous_predictions(data_dir):
    codec = ImageCodec.from_pretrained("polymathic-ai/aion-image-codec")

    input_batch_dict = torch.load(
        data_dir / "image_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_output = torch.load(
        data_dir / "image_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_output_tensor = torch.load(
        data_dir / "image_codec_decoded_batch.pt", weights_only=False
    )
    with torch.no_grad():
        print(input_batch_dict["image"]["channel_mask"][0])
        input_image_obj = Image(
            flux=input_batch_dict["image"]["array"][:, 5:],
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
        )
        encoded_output = codec.encode(input_image_obj)
        decoded_image_obj = codec.decode(
            encoded_output, bands=["DES-G", "DES-R", "DES-I", "DES-Z"]
        )

    # We flatten the reference encoded output to match the encoded output
    # as we now make all codecs return flattened outputs
    reference_encoded_output = reference_encoded_output.reshape(
        reference_encoded_output.shape[0], -1
    )

    assert encoded_output.shape == reference_encoded_output.shape
    assert torch.allclose(
        encoded_output,
        reference_encoded_output,
    )

    assert isinstance(decoded_image_obj, Image)
    assert torch.allclose(
        decoded_image_obj.flux,
        reference_decoded_output_tensor[:, 5:],
        rtol=1e-3,
        atol=1e-4,
    )
