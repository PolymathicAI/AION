import torch.nn.functional as F

from aion.modalities import Spectrum


def pad_spectrum(x: Spectrum) -> Spectrum:
    """Pad a Spectrum object to its specified pad_length.

    Note: Each of the sequence attributes (flux, ivar,
          mask, wavelength) should be 2D tensors.
    """
    padding_values = {"wavelength": 99999, "mask": True, "ivar": 0}

    for k in ["flux", "ivar", "mask", "wavelength"]:
        setattr(
            x,
            k,
            F.pad(
                getattr(x, k),
                (0, x.pad_length - getattr(x, k).shape[-1]),
                mode="constant",
                value=padding_values[k] if k in padding_values else 0,
            ),
        )

    return x
