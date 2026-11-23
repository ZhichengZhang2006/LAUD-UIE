"""Teacher utility measured by a virtual movement in RGB output space."""
import torch


def components(image):
    offset = image.mean(dim=(-2, -1), keepdim=True)
    return offset, image - offset


def component_distillation(prediction, taught, reference, gated):
    """Use a quarter step toward each teacher component to estimate its value."""
    losses, weights = [], []
    for student, teacher, target in zip(components(prediction), components(taught.detach()),
                                        components(reference.detach())):
        difference = teacher - student
        losses.append(difference.abs().flatten(1).mean(1))
        with torch.no_grad():
            error = student - target
            before = error.square().flatten(1).mean(1)
            after = (error + .25 * difference).square().flatten(1).mean(1)
            weights.append(((before - after) / before.clamp_min(1e-8)).clamp(0, 1))
    utility = torch.stack(weights, dim=1)
    active = utility if gated else torch.ones_like(utility)
    # Keep the total KD coefficient fixed when any component is useful.
    total = active.sum(1, keepdim=True)
    active = active / total.clamp_min(1e-12)
    return (active * torch.stack(losses, dim=1)).sum(1).mean(), active
