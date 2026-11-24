"""RGB enhancement from the supplied four-stage MambaCue feature backbone."""
import torch
from torch import nn


class SceneHead(nn.Module):
    def __init__(self, channels=32, centered=True):
        super().__init__()
        self.centered = centered
        self.spatial = nn.Conv2d(channels, 3, 1, bias=False)
        self.scene = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 8, 1),
                                   nn.SiLU(), nn.Conv2d(8, 3, 1))
        for layer in (self.spatial, self.scene[-1]):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.scene[-1].bias)

    def components(self, features):
        field = self.spatial(features)
        if self.centered:
            field = field - field.mean(dim=(-2, -1), keepdim=True)
        return self.scene(features), field

    def forward(self, image, features):
        offset, field = self.components(features)
        return (image + offset + field).clamp(0, 1)


class BackboneFusion(nn.Module):
    def __init__(self, centered=True):
        super().__init__()
        from .mambacue_backbone import MambaCueBackbone
        self.backbone = MambaCueBackbone(use_checkpoint=True)
        self.head = SceneHead(centered=centered)

    def forward(self, image):
        return self.head(image, self.backbone(image))

    def forward_bf16_backbone(self, image):
        """Run the feature backbone in autocast and reconstruct RGB in FP32."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = self.backbone(image)
        return self.head(image.float(), features.float())

    def set_inference_scan_batch(self, size):
        """Batch independent scans only during inference; weights are shared."""
        from .mambacue_backbone.encoder import FourDirectionMamba

        if size not in (1, 2, 4):
            raise ValueError('Scan batch must be 1, 2 or 4')
        for layer in self.backbone.modules():
            if isinstance(layer, FourDirectionMamba):
                layer.inference_scan_batch = size
