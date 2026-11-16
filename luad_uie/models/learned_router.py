"""Amortize same-image head alignment; this is not a generalization utility oracle."""
import torch
from torch import nn
from .gradient_router import HeadGradientRouter
from .utility_controller import Controller


class LearnedRouter(nn.Module):
    def __init__(self, student_dim, interval=10, warmup_steps=856):
        super().__init__()
        if interval <= 0 or warmup_steps < 0:
            raise ValueError('Invalid router probing schedule')
        self.interval, self.warmup_steps = interval, warmup_steps
        # Controller construction must not change augmentation/shuffle random streams.
        with torch.random.fork_rng(devices=[]):
            self.predictor = Controller(student_dim)
        self.teacher_router = HeadGradientRouter(interval=interval)

    def predict(self, degradation, student_state):
        return self.predictor(degradation, student_state).tanh()

    def weights(self, scores, step):
        scale = self.teacher_router.scale()
        positive = torch.ones_like(scores) if step < self.warmup_steps else scores.detach().clamp_min(0)
        weighted = positive * scale
        return (weighted / weighted.sum(1, keepdim=True).clamp_min(1e-12)).detach()

    def probe(self, prediction, spatial, reconstruction, views, step):
        return self.teacher_router(prediction, spatial, reconstruction, views, step)
