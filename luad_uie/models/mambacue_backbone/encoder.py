# Four-stage Mamba encoder copied from variant_04_stage_mamba.
import torch
from torch import nn
from mamba_ssm import Mamba
from torch.utils.checkpoint import checkpoint


class FourDirectionMamba(nn.Module):
    """
    Shared Mamba over row/column forward/reverse traversals.
    This is a four-traversal wrapper around mamba_ssm.Mamba, not the official
    VMamba SS2D implementation. Each direction is restored to BCHW before fusion.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, scan_mode="four"):
        super().__init__()
        if scan_mode not in ("row", "four"):
            raise ValueError("scan_mode must be 'row' or 'four'")
        self.scan_mode = scan_mode
        self.inference_scan_batch = 1
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Conv2d(dim, dim, 1)

    def _scan(self, x, column=False, reverse=False):
        grid = x.transpose(2, 3) if column else x
        batch, channels, height, width = grid.shape
        sequence = grid.flatten(2).transpose(1, 2).contiguous()
        if reverse:
            sequence = sequence.flip(1)
        sequence = self.mamba(sequence)
        if reverse:
            sequence = sequence.flip(1)
        grid = sequence.transpose(1, 2).reshape(batch, channels, height, width)
        return grid.transpose(2, 3).contiguous() if column else grid.contiguous()

    def _batched_scans(self, x):
        batch, channels, height, width = x.shape
        row = x.flatten(2).transpose(1, 2).contiguous()
        if self.inference_scan_batch == 2:
            row_sum = self._scan_pair(row).transpose(1, 2).reshape(batch, channels, height, width)
            del row
            col = x.transpose(2, 3).flatten(2).transpose(1, 2).contiguous()
            col_sum = self._scan_pair(col).transpose(1, 2).reshape(batch, channels, width, height)
            return ((row_sum + col_sum.transpose(2, 3)) * .25).contiguous()

        col = x.transpose(2, 3).flatten(2).transpose(1, 2).contiguous()
        inputs = (row, row.flip(1), col, col.flip(1))
        forward_row, backward_row, forward_col, backward_col = self.mamba(
            torch.cat(inputs, dim=0)).split(batch, dim=0)
        fused_row = forward_row + backward_row.flip(1)
        fused = fused_row.transpose(1, 2).reshape(batch, channels, height, width)
        forward = forward_col.transpose(1, 2).reshape(batch, channels, width, height)
        backward = backward_col.flip(1).transpose(1, 2).reshape(batch, channels, width, height)
        return ((fused + forward.transpose(2, 3) + backward.transpose(2, 3)) * 0.25).contiguous()

    def _scan_pair(self, sequence):
        forward, backward = self.mamba(torch.cat((sequence, sequence.flip(1)), dim=0)).chunk(2, dim=0)
        return forward + backward.flip(1)

    def forward(self, x):
        if (self.scan_mode == "four" and self.inference_scan_batch > 1
                and not self.training and not torch.is_grad_enabled()):
            return self.proj(self._batched_scans(x))
        fused = self._scan(x)
        if self.scan_mode == "four":
            # Sequential calls avoid stacking four full-resolution inputs.
            # Autograd still retains activations; stage checkpointing is optional.
            fused = fused + self._scan(x, reverse=True)
            fused = fused + self._scan(x, column=True)
            fused = fused + self._scan(x, column=True, reverse=True)
            fused = fused * 0.25
        return self.proj(fused)


class MambaBlock(nn.Module):
    """Local 2D residual followed by a channel-normalized Mamba residual."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, scan_mode="four"):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.norm = nn.LayerNorm(dim)
        self.context = FourDirectionMamba(dim, d_state, d_conv, expand, scan_mode)
        self.local_scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-3))
        self.context_scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-3))

    def forward(self, x):
        x = x + self.local_scale * self.local(x)
        normalized = self.norm(x.permute(0, 2, 3, 1))
        normalized = normalized.permute(0, 3, 1, 2).contiguous()
        return x + self.context_scale * self.context(normalized)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, stage_depths=(1, 1, 1, 1),
                 scan_mode="four", use_checkpoint=False):
        super().__init__()
        if len(stage_depths) != 4 or any(
            not isinstance(depth, int) or depth < 1 for depth in stage_depths
        ):
            raise ValueError("stage_depths must contain four positive integers")
        self.use_checkpoint = use_checkpoint
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.SiLU(),
        )
        channels = (base_ch, base_ch * 2, base_ch * 4, base_ch * 8)
        stages = [nn.Sequential(*[
            MambaBlock(dim, scan_mode=scan_mode) for _ in range(depth)
        ]) for dim, depth in zip(channels, stage_depths)]
        self.stage1, self.stage2, self.stage3, self.bottleneck = stages
        self.down1 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1)

    def _run_stage(self, stage, x):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, x):
        e1 = self._run_stage(self.stage1, self.stem(x))
        e2 = self._run_stage(self.stage2, self.down1(e1))
        e3 = self._run_stage(self.stage3, self.down2(e2))
        z = self._run_stage(self.bottleneck, self.down3(e3))
        return e1, e2, e3, z
