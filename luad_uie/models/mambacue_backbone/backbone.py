"""Standalone MambaCue feature backbone."""

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import Encoder


class ConvBlock(nn.Module):
    """Standard two-convolution U-Net block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        )
        self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        return self.block(torch.cat([x, skip], dim=1))


class BackboneDecoder(nn.Module):
    """The three U-Net decoder stages with skip fusion."""

    def __init__(self, base_ch=32):
        super().__init__()
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch, base_ch)

    def forward(self, e1, e2, e3, bottleneck):
        d3 = self.dec3(bottleneck, e3)
        d2 = self.dec2(d3, e2)
        return self.dec1(d2, e1)


class MambaCueBackbone(nn.Module):
    """Extract multiscale features and fuse them into a full-resolution map."""

    def __init__(
        self, in_channels=3, hidden_channels=32, stage_depths=(1, 1, 1, 1),
        scan_mode="four", use_checkpoint=False,
    ):
        super().__init__()
        if hidden_channels < 8 or hidden_channels % 8:
            raise ValueError("hidden_channels must be a positive multiple of 8")
        self.encoder = Encoder(
            in_ch=in_channels, base_ch=hidden_channels, stage_depths=stage_depths,
            scan_mode=scan_mode, use_checkpoint=use_checkpoint,
        )
        self.decoder = BackboneDecoder(hidden_channels)

    def forward(self, x):
        """Return a [B, hidden_channels, H, W] feature map."""
        return self.decoder(*self.encoder(x))
