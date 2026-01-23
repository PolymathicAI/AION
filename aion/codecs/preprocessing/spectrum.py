import torch.nn.functional as F

from aion.modalities import Spectrum


def pad_spectrum(x: Spectrum):
    padding_values = {"lambda": 99999, "mask": True, "ivar": 0}

    for k in ["flux", "ivar", "mask", "wavelength"]:
        x.k = F.pad(
            x.k,
            (0, x.pad_length - len(x.k)),
            mode="constant",
            value=padding_values[k] if k in padding_values else 0,
        )

    return x
