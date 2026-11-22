"""A single replacement output head for structure-preserving restoration."""
import torch
from torch import nn
from torch.nn import functional as F
from .student import Student
from .packed_student import PackedStudent


def local_mean(x):
    return F.avg_pool2d(x, 5, stride=1, padding=2, count_include_pad=False)


class SelectiveStudent(Student):
    def __init__(self, width=24, gated=True):
        super().__init__(width)
        self.gated = gated
        # Four residual controls: low-frequency luma, two opponent colors, detail gain.
        self.head = nn.Conv2d(width, 7 if gated else 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, return_features=False):
        height, width = x.shape[-2:]
        small = F.interpolate(x, size=(max(4, (height+1)//2), max(4, (width+1)//2)), mode='area')
        a = self.enc1(small)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.bottleneck(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        f = self.dec1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        controls = F.interpolate(self.head(f), size=(height,width), mode='bilinear', align_corners=False)
        light, red, blue, detail = controls[:, :4].tanh().split(1, dim=1)
        if self.gated:
            intensity_gate, color_gate, detail_gate = controls[:, 4:].sigmoid().split(1, dim=1)
            light, red, blue, detail = light*intensity_gate, red*color_gate, blue*color_gate, detail*detail_gate
        # Opponent correction has zero luma before gamut clipping.
        green = -(.2126*red+.0722*blue)/.7152
        luma = .2126*x[:, :1]+.7152*x[:, 1:2]+.0722*x[:, 2:3]
        correction = detail*(luma-local_mean(luma))
        restored = (x+local_mean(light)+torch.cat([red,green,blue],dim=1)
                    + correction-local_mean(correction)).clamp(0,1)
        if return_features:
            return restored, {'pooled':f.mean((2,3)), 'spatial':f}
        return restored


def build_student(config):
    name = config.get('architecture', 'unet')
    if name == 'unet':
        return Student(config['width'])
    if name == 'selective':
        return SelectiveStudent(config['width'], gated=config.get('gated', True))
    if name == 'packed':
        return PackedStudent(config['width'], phase_residual=config.get('phase_residual', True),
                             global_context=config.get('global_context', False))
    raise ValueError(f'Unknown student architecture: {name}')
