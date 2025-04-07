import torch


def to(x, device='cuda'):
    def _move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    return torch.utils._pytree.tree_map(_move, x)
