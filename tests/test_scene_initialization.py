import torch

from luad_uie.models.backbone_fusion import SceneHead
from luad_uie.models.scene_initialization import fit_components, initialize_head


def test_calibrated_heads_recover_same_unseen_corrections():
    random = torch.Generator().manual_seed(8)
    def normal(*shape):
        return torch.randn(shape, generator=random)
    channels = 8
    scene, spatial, bias = normal(3, channels) * .02, normal(3, channels) * .02, normal(3) * .03
    def sample():
        field = normal(1, channels, 9, 11) + normal(1, channels, 1, 1)
        mean = field.mean((2, 3), keepdim=True)
        correction = (torch.einsum('oc,bchw->bohw', scene, mean) + bias[None,:,None,None]
                      + torch.einsum('oc,bchw->bohw', spatial, field - mean))
        return field, correction
    coefficients = fit_components([sample() for _ in range(32)], regularization=1e-8)
    field, expected = sample()
    for centered in (False, True):
        head = SceneHead(channels, centered=centered)
        initialize_head(head, coefficients)
        offset, delta = head.components(field)
        torch.testing.assert_close(offset + delta, expected, atol=1e-6, rtol=1e-5)
        (offset + delta).square().mean().backward()
        assert head.spatial.weight.grad.abs().sum() > 0
        assert head.scene[1].weight.grad.abs().sum() > 0
