import torch
from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX, BAND_CENTER_MAX


class ImagePadder:
    """Formatter that pads the images to have a fixed number of bands."""

    def __init__(self):
        self.nbands = max(BAND_TO_INDEX.values()) + 1

    def _check_bands(self, bands: list[str]):
        for band in bands:
            if band not in BAND_TO_INDEX:
                raise ValueError(
                    f"Invalid band: {band}. Valid bands are: {list(BAND_TO_INDEX.keys())}"
                )

    def forward(self, image, bands):
        num_channels = self.nbands
        batch, _, height, width = image.shape

        # Check if bands are valid
        self._check_bands(bands)

        # Create a new image array with the correct number of channels
        padded_image = torch.zeros(
            (batch, num_channels, height, width), dtype=image.dtype
        ).to(image.device)

        # Create a list of new channel indices based on the order of bands
        new_channel_indices = [
            BAND_TO_INDEX[band] for band in bands if band in BAND_TO_INDEX
        ]

        # Vectorized assignment of the original channels to the new positions
        padded_image[:, new_channel_indices, :, :] = image[
            :, : len(new_channel_indices), :, :
        ]

        # Get boolean mask of channels that are present
        channel_mask = torch.zeros(num_channels, dtype=torch.bool).to(image.device)
        channel_mask[new_channel_indices] = True
        channel_mask = channel_mask.unsqueeze(0).expand(batch, -1)
        return padded_image, channel_mask

    def backward(self, padded_image, bands):
        # Check if bands are valid
        self._check_bands(bands)

        # Get the indices for the requested bands
        channel_indices = [BAND_TO_INDEX[b] for b in bands]

        # Select those channels along dim=1
        selected_image = padded_image[:, channel_indices, :, :]
        return selected_image


class CenterCrop:
    """Formatter that crops the images to have a fixed number of bands."""

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, image):
        _, _, height, width = image.shape
        start_x = (width - self.crop_size) // 2
        start_y = (height - self.crop_size) // 2
        return image[
            :, :, start_y : start_y + self.crop_size, start_x : start_x + self.crop_size
        ]


class Clamp:
    """Formatter that clamps the images to a given range."""

    def __init__(self):
        self.clamp_dict = BAND_CENTER_MAX

    def __call__(self, image, bands):
        for i, band in enumerate(bands):
            image[:, i, :, :] = torch.clip(
                image[:, i, :, :], -self.clamp_dict[band], self.clamp_dict[band]
            )
        return image


class RescaleToLegacySurvey:
    """Formatter that rescales the images to have a fixed number of bands."""

    def __init__(self):
        pass

    def convert_zeropoint(self, zp: float) -> float:
        return 10.0 ** ((zp - 22.5) / 2.5)

    def reverse_zeropoint(self, scale: float) -> float:
        return 22.5 - 2.5 * torch.log10(scale)

    def forward(self, image, survey):
        zpscale = self.convert_zeropoint(27.0) if survey == "HSC" else 1.0
        image /= zpscale
        return image

    def backward(self, image, survey):
        zpscale = self._reverse_zeropoint(27.0) if survey == "HSC" else 1.0
        image *= zpscale
        return image
