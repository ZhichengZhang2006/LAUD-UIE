import torch
from torch import nn
from torch.nn import functional as F


def ssim(x, y):
    """Per-image RGB SSIM, Gaussian 11x11 sigma=1.5, valid window, data_range=1."""
    size = min(11, x.shape[-2], x.shape[-1])
    size -= 1 - size % 2
    coord = torch.arange(size, device=x.device, dtype=x.dtype) - (size - 1) / 2
    g = torch.exp(-coord.square() / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(x.shape[1], 1, size, size)
    def avg(t):
        return F.conv2d(t, kernel, groups=x.shape[1])
    mx, my = avg(x), avg(y)
    vx, vy = (avg(x*x)-mx*mx).clamp_min(0), (avg(y*y)-my*my).clamp_min(0)
    cov = avg(x*y)-mx*my
    score = ((2*mx*my+0.01**2)*(2*cov+0.03**2)) / ((mx*mx+my*my+0.01**2)*(vx+vy+0.03**2))
    return score.mean((1, 2, 3))


class ReconstructionLoss(nn.Module):
    def __init__(self, ssim_weight=0.2, epsilon=1e-3):
        super().__init__()
        self.ssim_weight, self.epsilon = ssim_weight, epsilon

    def forward(self, prediction, target):
        char = ((prediction-target).square()+self.epsilon**2).sqrt().mean((1, 2, 3))
        return (char + self.ssim_weight*(1-ssim(prediction, target))).mean()


def metrics(prediction, target):
    mse = (prediction-target).square().mean((1, 2, 3))
    return {'mse': mse, 'psnr': -10*torch.log10(mse.clamp_min(1e-12)), 'ssim': ssim(prediction, target)}
