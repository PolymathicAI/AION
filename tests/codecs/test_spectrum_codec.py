import torch

from aion.modalities import Spectrum
from aion.codecs import SpectrumCodec


def test_hf_previous_predictions(data_dir):
    codec = SpectrumCodec.from_pretrained("polymathic-ai/aion-spectrum-codec")

    input_batch = torch.load(data_dir / "SPECTRUM_input_batch.pt", weights_only=False)[
        "spectrum"
    ]
    reference_encoded_output = torch.load(
        data_dir / "SPECTRUM_encoded_batch.pt", weights_only=False
    )
    reference_decoded_output = torch.load(
        data_dir / "SPECTRUM_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        # Create Spectrum modality instance
        spectrum_input = Spectrum(
            flux=input_batch["flux"],
            ivar=input_batch["ivar"],
            mask=input_batch["mask"],
            wavelength=input_batch["lambda"],
        )

        encoded_output = codec.encode(spectrum_input)
        assert encoded_output.shape == reference_encoded_output.shape
        assert torch.allclose(encoded_output, reference_encoded_output)

        decoded_spectrum = codec.decode(encoded_output)

        assert (
            decoded_spectrum.flux.shape
            == reference_decoded_output["spectrum"]["flux"].shape
        )
        assert torch.allclose(
            decoded_spectrum.flux,
            reference_decoded_output["spectrum"]["flux"],
            rtol=1e-3,
            atol=1e-4,
        )
        assert (
            decoded_spectrum.wavelength.shape
            == reference_decoded_output["spectrum"]["lambda"].shape
        )
        assert torch.allclose(
            decoded_spectrum.wavelength,
            reference_decoded_output["spectrum"]["lambda"],
            rtol=1e-3,
            atol=1e-4,
        )
        assert (
            decoded_spectrum.mask.shape
            == reference_decoded_output["spectrum"]["mask"].shape
        )
        assert torch.allclose(
            decoded_spectrum.mask, reference_decoded_output["spectrum"]["mask"].bool()
        )
