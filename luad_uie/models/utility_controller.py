import torch
from torch import nn


class Controller(nn.Module):
    def __init__(self, student_dim, hidden=64, threshold=0.):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(20+student_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.threshold = threshold

    def forward(self, degradation_state, student_state):
        return self.network(torch.cat([degradation_state.detach(), student_state.detach()], 1))

    def weights(self, utility):
        positive = (utility.detach()-self.threshold).clamp_min(0)
        return positive / positive.sum(1, keepdim=True).clamp_min(1e-12)
