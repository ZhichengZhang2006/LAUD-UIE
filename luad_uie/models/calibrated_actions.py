import torch
from torch import nn
from .degradation import controller_inputs


def action_context(x, output, features, progress, lr):
    d, s = controller_inputs(x, output, features)
    z = torch.cat([d, s], 1)
    return torch.cat([z, z.mean(0, keepdim=True).expand_as(z),
                      z.new_tensor([progress, lr/.0002]).expand(len(z), 2)], 1).detach()


class CalibratedActionPredictor(nn.Module):
    def __init__(self, student_dim):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.net = nn.Sequential(nn.Linear(2*(20+student_dim)+2, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 7*2*4*3))

    def forward(self, context):
        raw = self.net(context.detach()).reshape(-1, 7, 2, 4, 3)
        # Monotone quantiles, initialized with modest interval widths.
        middle = raw[..., 0]
        return torch.stack([middle-raw[..., 1].sigmoid()*.1, middle,
                            middle+raw[..., 2].sigmoid()*.1], -1)

    @staticmethod
    def loss(prediction, target):
        residual = target[..., None]-prediction
        tau = prediction.new_tensor([.1, .5, .9])
        return torch.maximum(tau*residual, (tau-1)*residual).mean()


class UtilityBuffer:
    def __init__(self, capacity=256):
        self.capacity, self.samples = capacity, []

    def add(self, context, gain, step):
        self.samples.append((context.detach().cpu(), gain.detach().cpu(), step))
        self.samples = self.samples[-self.capacity:]

    def recent(self, step, max_age=10000):
        return [s for s in self.samples if step-s[2] <= max_age]


def lower_bound(predictor, prediction, calibration, scale, step, min_samples=8):
    samples = calibration.recent(step)
    if len(samples) < min_samples:
        return None
    device = prediction.device
    with torch.no_grad():
        contexts = torch.cat([x for x, _, _ in samples]).to(device)
        truth = torch.stack([y for _, y, _ in samples]).to(device)/scale
        residual = predictor(contexts)[..., 0]-truth
        # Empirical marginal 90% correction; no simultaneous safety guarantee.
        margin = torch.quantile(residual, .9, dim=0).clamp_min(0)
        return prediction[..., 0]-margin


def choose_actions(prediction, lower=None, risk=False, warmup=False):
    if warmup or (risk and lower is None):
        return torch.zeros(len(prediction), dtype=torch.long, device=prediction.device)
    score = lower if risk else prediction[..., 1]
    if risk:
        feasible = (score[..., 0] > 0).all(-1) & (score[..., 1:] >= -.01).all(-1).all(-1)
    else:
        feasible = score[..., 0].mean(-1) > 0
    value = score[..., 0].mean(-1).masked_fill(~feasible, -float('inf'))
    best = value.argmax(1)
    return torch.where(feasible.any(1), best+1, torch.zeros_like(best))


class ProbeScheduler:
    def __init__(self, active=False, interval=100):
        self.active, self.interval = active, interval
        self.credit, self.count, self.last = 0., 0, -1

    def due(self, step, uncertain=True):
        self.credit = min(5., self.credit+1/self.interval)
        periodic = (step+1) % self.interval == 0
        forced = (step+1) % (5*self.interval) == 0
        decision = periodic if not self.active else forced or (uncertain and step % 20 == 0)
        if decision and self.credit >= 1-1e-9:
            self.credit = max(0., self.credit-1)
            self.count += 1
            self.last = step
            return True
        return False
