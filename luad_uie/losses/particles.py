"""Paired synthetic particle perturbations for structure-aware distillation."""
import torch
from ..models.knowledge_views import luminance


@torch.no_grad()
def add_particles(image, seed, count=20):
    """Create an aligned bright scattering perturbation and its support."""
    generator = torch.Generator(device=image.device).manual_seed(seed)
    batch, _, height, width = image.shape
    def uniform(shape):
        return torch.rand(shape, device=image.device, generator=generator, dtype=image.dtype)

    support = image.new_zeros((batch, 1, height, width))
    yy = torch.arange(height, device=image.device)[None, None, :, None]
    xx = torch.arange(width, device=image.device)[None, None, None, :]
    centers = uniform((batch, count, 2))
    sigma = (.55 + 1.75 * uniform((batch, count, 1, 1)))
    amplitudes = .12 + .45 * uniform((batch, count, 1, 1))
    # Chunk components to avoid an unbounded full-resolution batch tensor.
    for offset in range(0, count, 4):
        indices = slice(offset, offset + 4)
        dy = yy - centers[:, indices, 0, None, None] * (height - 1)
        dx = xx - centers[:, indices, 1, None, None] * (width - 1)
        radius = (dx.square() + dy.square()) / sigma[:, indices].square()
        opacity = amplitudes[:, indices] * (-.5 * radius).exp()
        opacity = opacity.masked_fill(radius > 9, 0)
        support = torch.maximum(support, opacity.amax(1, keepdim=True))
    return image + support * (1 - image), support > 0


def gradient_pairs(image):
    y = luminance(image)
    return y[..., 1:] - y[..., :-1], y[..., 1:, :] - y[..., :-1, :]


def stable_structure_loss(original, perturbed, teacher, reference, mask):
    """Suppress responses to new particles and match trusted reference edges."""
    axes = (
        mask[..., 1:] | mask[..., :-1],
        mask[..., 1:, :] | mask[..., :-1, :],
    )
    parts = []
    for inside, base, changed, known, truth in zip(
            axes, gradient_pairs(original), gradient_pairs(perturbed),
            gradient_pairs(teacher.detach()), gradient_pairs(reference)):
        # Supervise the mask with the reference. Distill a teacher edge outside
        # the mask only when the paired target corroborates it.
        reliable = (~inside) & (truth.abs() >= .02) & ((known - truth).abs() <= .03)
        error = torch.where(inside, (changed - truth).abs(), (changed - known).abs())
        selected = inside | reliable
        invariance = (changed - base.detach()).abs()
        parts.append((error * selected).sum() / selected.sum().clamp_min(1)
                     + .25 * (invariance * inside).sum() / inside.sum().clamp_min(1))
    return sum(parts) / len(parts)
