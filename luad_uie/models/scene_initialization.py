"""Fit the scene and local corrections of the existing head on paired training data."""
import torch


def _solve(gram, cross, regularization):
    diagonal = gram.diag().mean().clamp_min(1e-8) * regularization
    system = gram + torch.eye(gram.size(0), dtype=gram.dtype) * diagonal
    return torch.linalg.solve(system, cross).T


@torch.no_grad()
def fit_components(feature_pairs, regularization=1e-3):
    """Each pair contains one decoder feature map and one reference-minus-input image."""
    means, offsets = [], []
    gram = cross = None
    for index, (features, target) in enumerate(feature_pairs):
        field = features.squeeze(0).flatten(1)
        correction = target.squeeze(0).flatten(1)
        mean = field.mean(1)
        offset = correction.mean(1)
        means.append(mean.double().cpu())
        offsets.append(offset.double().cpu())
        # Sample up to 256 positions per training image; use full-image means for centering.
        generator = torch.Generator().manual_seed(1147 + index)
        positions = torch.randperm(field.size(1), generator=generator)[:256].to(field.device)
        design = (field[:, positions] - mean[:, None]).double().cpu()
        response = (correction[:, positions] - offset[:, None]).double().cpu()
        moment = design @ design.T / positions.numel()
        mixed = design @ response.T / positions.numel()
        gram = moment if gram is None else gram + moment
        cross = mixed if cross is None else cross + mixed
    if len(means) < 2:
        raise ValueError('Fit the scene correction on at least two training images')
    x, y = torch.stack(means), torch.stack(offsets)
    xm, ym = x.mean(0), y.mean(0)
    x, y = x - xm, y - ym
    scene = _solve(x.T @ x, x.T @ y, regularization)
    spatial = _solve(gram, cross, regularization)
    return scene, ym - scene @ xm, spatial


@torch.no_grad()
def initialize_head(head, coefficients):
    """Both head parameterizations produce the same pre-clipping RGB correction."""
    scene, bias, spatial = coefficients
    slope = scene if head.centered else scene - spatial
    head.spatial.weight.copy_(spatial[:, :, None, None])
    first, last = head.scene[1], head.scene[-1]
    # SiLU(z) - SiLU(-z) = z. Keep the last two hidden units trainable at their
    # original random input weights with zero output weights.
    first.weight[:3].copy_(slope[:, :, None, None])
    first.weight[3:6].copy_(-slope[:, :, None, None])
    first.bias[:6].zero_()
    last.weight.zero_()
    identity = torch.eye(3, dtype=last.weight.dtype, device=last.weight.device)
    last.weight[:, :3, 0, 0].copy_(identity)
    last.weight[:, 3:6, 0, 0].copy_(-identity)
    last.bias.copy_(bias)
