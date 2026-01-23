import numpy as np


def pad_spectrum(sample, max_length: int = 4800):
    padding_values = {"lambda": 99999, "mask": True, "ivar": 0}

    for k in sample["spectrum"].keys():
        sample["spectrum"][k] = np.pad(
            sample["spectrum"][k],
            (0, max_length - len(sample["spectrum"][k])),
            constant_values=padding_values[k] if k in padding_values else 0,
        )

    return sample
