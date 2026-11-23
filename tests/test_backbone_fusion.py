import torch

from luad_uie.losses.component_distillation import component_distillation
from luad_uie.losses.ordinal_structure import ordinal_structure
from luad_uie.models.backbone_fusion import SceneHead


def test_fusion_preserves_input_and_centers_after_learning():
    head = SceneHead()
    image = torch.rand(2, 3, 17, 19)
    features = torch.randn(2, 32, 17, 19, requires_grad=True)
    assert torch.equal(head(image, features), image)
    torch.nn.init.normal_(head.spatial.weight, std=.01)
    _, field = head.components(features)
    assert torch.allclose(field.mean((2, 3)), torch.zeros(2, 3), atol=1e-6)
    result = head(image, features)
    loss = (result - .6).square().mean() + ordinal_structure(result, image, image)
    loss.backward()
    assert head.spatial.weight.grad.abs().sum() > 0
    assert head.scene[-1].weight.grad.abs().sum() > 0
    assert features.grad.abs().sum() > 0


def test_component_gate_accepts_offset_rejects_false_texture():
    prediction = torch.full((1, 3, 8, 8), .4, requires_grad=True)
    target = torch.full_like(prediction, .6)
    teacher = target.clone()
    teacher[..., ::2] += .1
    teacher[..., 1::2] -= .1
    teacher.requires_grad_(True)
    loss, weight = component_distillation(prediction, teacher, target, True)
    assert torch.allclose(weight, torch.tensor([[1., 0.]]))
    loss.backward()
    assert prediction.grad.abs().sum() > 0 and teacher.grad is None
    fixed, equal = component_distillation(prediction, teacher, target, False)
    assert torch.equal(equal, torch.tensor([[.5, .5]]))
    assert torch.isfinite(fixed)


def test_no_teacher_signal_when_target_already_met():
    reference = torch.randn(2, 3, 8, 8)
    student = reference.clone().requires_grad_()
    loss, weight = component_distillation(student, reference + .2, reference, True)
    assert torch.count_nonzero(weight) == 0
    loss.backward()
    assert torch.count_nonzero(student.grad) == 0
