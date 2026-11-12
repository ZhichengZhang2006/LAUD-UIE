"""Sample-specific distillation routing at a sigmoid RGB 1x1 reconstruction head."""
import torch
from torch import nn


class HeadGradientRouter(nn.Module):
    def __init__(self, alignment=True, decay=.9, interval=10):
        super().__init__()
        self.alignment = alignment
        self.decay = decay
        self.interval = interval
        self.register_buffer("ema_norm", torch.zeros(3))

    @staticmethod
    def head_vectors(loss, prediction, spatial):
        # Batch-separable losses are summed before this call. No .grad buffers are written.
        rgb_grad, = torch.autograd.grad(loss, prediction, retain_graph=True)
        with torch.no_grad():
            logits_grad = (rgb_grad * prediction * (1 - prediction)).flatten(2)
            weight = torch.bmm(logits_grad, spatial.detach().flatten(2).transpose(1, 2))
            bias = logits_grad.sum(2)
            return torch.cat((weight.flatten(1), bias), dim=1)

    def forward(self, prediction, spatial, reconstruction, views, step):
        update = step % self.interval == 0
        if self.alignment or update:
            vectors = torch.stack([
                self.head_vectors(loss.sum(), prediction, spatial)
                for loss in views.unbind(1)
            ], dim=1)
            if update:
                norms = vectors.norm(dim=2).mean(0)
                if step == 0:
                    self.ema_norm.copy_(norms)
                else:
                    self.ema_norm.lerp_(norms, 1 - self.decay)
        scale = ((self.ema_norm.median() + 1e-12) /
                 (self.ema_norm + 1e-12)).clamp(.1, 10)
        if self.alignment:
            recovery = self.head_vectors(reconstruction * len(prediction), prediction, spatial)
            cosine = torch.nn.functional.cosine_similarity(vectors, recovery[:, None], dim=2, eps=1e-12)
            score = cosine.clamp_min(0) * scale
        else:
            cosine = None
            score = scale.expand_as(views)
        weights = score / score.sum(1, keepdim=True).clamp_min(1e-12)
        return weights.detach(), cosine
