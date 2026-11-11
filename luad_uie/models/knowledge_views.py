import torch
from torch import nn
from torch.nn import functional as F


def luminance(x):
    return (x * x.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)


def lab_chroma(x):
    """Differentiable sRGB -> D65 CIE Lab a/b, divided by 128."""
    rgb = torch.where(x > 0.04045, ((x+0.055)/1.055).clamp_min(1e-8).pow(2.4), x/12.92)
    matrix = x.new_tensor([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, rgb)
    xyz = xyz / x.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    delta = 6/29
    f = torch.where(xyz > delta**3, xyz.clamp_min(1e-8).pow(1/3), xyz/(3*delta**2)+4/29)
    return torch.stack([500*(f[:, 0]-f[:, 1]), 200*(f[:, 1]-f[:, 2])], 1)/128


class KnowledgeView(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name not in ('color', 'illumination', 'detail'):
            raise ValueError(name)
        self.name = name

    def forward(self, student, teacher):
        teacher = teacher.detach()
        if self.name == 'color':
            return (lab_chroma(student)-lab_chroma(teacher)).abs().mean((1, 2, 3))
        s, t = luminance(student), luminance(teacher)
        if self.name == 'illumination':
            s = F.avg_pool2d(s, 9, 1, 4, count_include_pad=False)
            t = F.avg_pool2d(t, 9, 1, 4, count_include_pad=False)
            return (s-t).abs().mean((1, 2, 3))
        dx = ((s[..., 1:]-s[..., :-1])-(t[..., 1:]-t[..., :-1])).abs().mean((1, 2, 3))
        dy = ((s[..., 1:, :]-s[..., :-1, :])-(t[..., 1:, :]-t[..., :-1, :])).abs().mean((1, 2, 3))
        return (dx+dy)/2


class KnowledgeViews(nn.Module):
    names = ('color', 'illumination', 'detail')

    def __init__(self):
        super().__init__()
        self.views = nn.ModuleList([KnowledgeView(n) for n in self.names])

    def forward(self, student, teacher):
        """Return [B, 3] so weights and losses remain sample-specific."""
        return torch.stack([view(student, teacher) for view in self.views], 1)
