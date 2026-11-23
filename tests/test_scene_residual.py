import pytest
import torch

from luad_uie.models.selective_student import build_student
from luad_uie.scripts.train_matched_quality import scene_metrics, scene_supervision, student_config


@pytest.mark.parametrize('shape', [(16, 18), (15, 17), (3, 5)])
def test_centered_residual_uses_visible_domain(shape):
    torch.manual_seed(11)
    network = build_student(student_config('scene_supervised', 4))
    image = torch.rand(2, 3, *shape) * .4 + .3
    assert torch.equal(network(image), image)
    with torch.no_grad():
        network.head.weight.normal_(0, .03)
        network.head.bias.copy_(torch.linspace(-.15, .15, 15))
    output, features = network(image, return_features=True)
    spatial = features['spatial_delta']
    assert spatial.shape == image.shape
    assert spatial.abs().max() > 0
    assert torch.allclose(spatial.mean((-2, -1)), torch.zeros(2, 3), atol=1e-7)
    assert torch.allclose((output - image).mean((-2, -1), keepdim=True), features['scene_offset'], atol=1e-6)
    scene_supervision(features, image, image + .05).backward()
    assert torch.isfinite(network.head.weight.grad).all()
    assert network.head.weight.grad[:12].abs().sum() > 0
    assert network.head.weight.grad[12:].abs().sum() > 0


def test_scene_control_matches_capacity_and_metrics_partition():
    centered = build_student(student_config('scene_centered', 4))
    free = build_student(student_config('scene_unconstrained', 4))
    with torch.no_grad():
        centered.head.bias[:12].fill_(.1)
    free.load_state_dict(centered.state_dict())
    image = torch.full((1, 3, 16, 16), .4)
    assert torch.allclose(centered(image), image, atol=1e-7)
    assert (free(image) > image).all()
    output = torch.rand_like(image)
    score = scene_metrics(output, image)
    assert torch.allclose(score['scene_mse'] + score['spatial_mse'], (output - image).square().mean())
