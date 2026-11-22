"""Half-resolution enhancement with a lossless input and a signed phase residual."""

import torch
from torch import nn
from torch.nn import functional as F

from .student import Student, block


class PackedStudent(Student):
    """Predict an RGB correction for each pixel in a 2×2 input cell."""

    def __init__(self, width=24, phase_residual=True):
        super().__init__(width)
        self.enc1 = block(12, width)
        self.phase_residual = phase_residual
        # Match parameter budget when ablating the centered residual.
        self.head = nn.Conv2d(width, 12, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, return_features=False):
        height, width = x.shape[-2:]
        padded = F.pad(x, (0, max(8-width, width % 2), 0, max(8-height, height % 2)), mode='replicate')
        packed = F.pixel_unshuffle(padded, 2)
        e1 = self.enc1(packed)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        bridge = self.bottleneck(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(bridge, size=e2.shape[-2:], mode='bilinear',
                                               align_corners=False), e2], dim=1))
        features = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear',
                                                     align_corners=False), e1], dim=1))
        correction = self.head(features).tanh().unflatten(1, (3, 4))
        mean = correction.mean(dim=2)
        smooth = F.interpolate(mean, size=padded.shape[-2:], mode='bilinear', align_corners=False)
        if self.phase_residual:
            centered = correction - mean.unsqueeze(2)
            smooth = smooth + F.pixel_shuffle(centered.flatten(1, 2), 2)
        result = (padded + smooth).clamp(0, 1)[..., :height, :width]
        if return_features:
            return result, {'pooled': features.mean((2, 3)), 'spatial': features}
        return result
