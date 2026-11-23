"""Training-only structure loss based on aligned raw/reference luminance order."""
from torch.nn import functional as F

from ..models.knowledge_views import luminance


OFFSETS = (1, 2, 4)
MIN_CONTRAST = .02
MARGIN_FRACTION = .5


def differences(image):
    """Signed endpoint differences at three separations in each spatial direction."""
    y = luminance(image)
    for stride in OFFSETS:
        if stride < y.shape[-1]:
            yield y[..., stride:] - y[..., :-stride]
        if stride < y.shape[-2]:
            yield y[..., stride:, :] - y[..., :-stride, :]


def consensus_terms(prediction, image, reference):
    """Positive margins supported by two aligned views of the same scene."""
    for actual, observed, target in zip(differences(prediction), differences(image), differences(reference)):
        contrast = observed.abs().minimum(target.abs())
        supported = (contrast >= MIN_CONTRAST) & (observed.sign() == target.sign())
        directed = actual * target.sign()
        yield F.relu(MARGIN_FRACTION * contrast - directed), supported


def ordinal_structure(prediction, image, reference):
    loss = prediction.new_zeros(())
    support = prediction.new_zeros(())
    for violation, mask in consensus_terms(prediction, image.detach(), reference.detach()):
        loss = loss + (violation * mask).sum()
        support = support + mask.sum()
    return loss / support.clamp_min(1)


def full_gradient(prediction, reference):
    absolute = [(actual - target).abs().mean()
                for actual, target in zip(differences(prediction), differences(reference))]
    return sum(absolute, prediction.new_zeros(())) / max(1, len(absolute))


def structure_metrics(prediction, reference, image):
    batch = prediction.shape[0]
    errors = prediction.new_zeros(batch)
    selected = prediction.new_zeros(batch)
    total = 0
    for violation, mask in consensus_terms(prediction, image, reference):
        errors = errors + ((violation > 0) & mask).flatten(1).sum(1)
        selected = selected + mask.flatten(1).sum(1)
        total += mask[0].numel()
    return {'ordinal_violation_rate': errors / selected.clamp_min(1),
            'ordinal_anchor_fraction': selected / max(1, total)}
