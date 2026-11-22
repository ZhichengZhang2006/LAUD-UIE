"""Training-only paired-shift supervision for edges crossing pixel-packing boundaries."""

import torch
from torch.nn import functional as F


def shift(image, dx, dy):
    height, width = image.shape[-2:]
    return F.pad(image, (dx, 0, dy, 0), mode='replicate')[..., :height, :width]


def interior(image, dx=0, dy=0, border=32):
    height, width = image.shape[-2:]
    return image[..., border+dy:height-border+dy, border+dx:width-border+dx]


def phase_edge_loss(original, translated, target, dx, dy):
    """Match signed edge responses after restoring the original coordinate system."""
    base = interior(original)
    moved = interior(translated, dx, dy)
    reference = interior(target)
    penalty = base.new_zeros(())
    count = base.new_zeros(())
    for axis in (2, 3):
        signal = torch.diff(reference, dim=axis)
        edges = signal.abs().amax(dim=1, keepdim=True) >= .02
        # A fixed reference determines edge support, so the network cannot reduce
        # the cost by deleting its own edges. RGB supervision remains separate.
        disagreement = torch.diff(moved - base.detach(), dim=axis).abs().mean(dim=1, keepdim=True)
        penalty = penalty + (disagreement * edges).sum()
        count = count + edges.sum()
    return penalty / count.clamp_min(1)
