import torch
import torch.nn.functional as F


def pad_spectrum(sample, max_length: int = 4800):
    padding_values = {"lambda": 99999, "mask": True, "ivar": 0}

    for k in sample["spectrum"].keys():
        if (
            isinstance(sample["spectrum"][k], torch.Tensor)
            and sample["spectrum"][k].ndim == 1
        ):
            sample["spectrum"][k] = F.pad(
                sample["spectrum"][k],
                (0, max_length - len(sample["spectrum"][k])),
                mode="constant",
                value=padding_values[k] if k in padding_values else 0,
            )

    return sample
