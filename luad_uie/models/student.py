import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


class Student(nn.Module):
    """Small U-Net; arbitrary H/W >= 4; RGB output in [0, 1]."""
    def __init__(self, width=24):
        super().__init__()
        self.enc1 = block(3, width)
        self.enc2 = block(width, width * 2)
        self.bottleneck = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, 3, 1)
        self.feature_dim = width

    def forward(self, x, return_features=False):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.bottleneck(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        f = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        out = torch.sigmoid(self.head(f))
        return (out, {"pooled": f.mean((2, 3)), "spatial": f}) if return_features else out

    def utility_parameters(self):
        """Replacement backbones must expose a small reconstruction parameter subset."""
        return dict(self.head.named_parameters(prefix='head'))
