import torch
from .knowledge_views import luminance


def image_statistics(x):
    """10 fixed features: RGB means/stds, luma mean/std, horizontal/vertical contrast."""
    y = luminance(x)
    return torch.cat([x.mean((2, 3)), x.std((2, 3), unbiased=False),
                      y.mean((2, 3)), y.std((2, 3), unbiased=False),
                      (y[..., 1:]-y[..., :-1]).abs().mean((2, 3)),
                      (y[..., 1:, :]-y[..., :-1, :]).abs().mean((2, 3))], 1)


def controller_inputs(x, output, features):
    return image_statistics(x).detach(), torch.cat([features['pooled'], image_statistics(output)], 1).detach()
