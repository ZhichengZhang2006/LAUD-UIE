import copy
import torch
import pytest

from luad_uie.models import Student, Teacher, KnowledgeViews, Controller
from luad_uie.models.degradation import controller_inputs
from luad_uie.losses import ReconstructionLoss, metrics
from luad_uie.datasets import validate_splits
from luad_uie.trainers.utility import estimate_utility
from luad_uie.trainers.engine import strategy_weights, validate


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(10)
    torch.set_num_threads(1)


def test_student_shapes_and_views_gradient():
    s = Student(4)
    x = torch.rand(2, 3, 17, 19)
    y, features = s(x, True)
    assert y.shape == x.shape and features['pooled'].shape == (2, 4)
    views = KnowledgeViews()(y, x)
    assert views.shape == (2, 3)
    assert torch.equal(KnowledgeViews()(x, x), torch.zeros(2, 3))
    (views.sum()+ReconstructionLoss()(y, x)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in s.parameters())


def test_teacher_frozen_and_eval():
    teacher = Teacher(Student(4))
    before = copy.deepcopy(teacher.state_dict())
    teacher.train()
    output = teacher(torch.rand(1, 3, 16, 16, requires_grad=True))
    assert not teacher.training and not teacher.model.training and not output.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v, before[k]) for k, v in teacher.state_dict().items())


def test_no_distillation_and_controller_detach():
    controller = Controller(4)
    utility = torch.tensor([[-1., 0., -2.], [1., -2., 3.]])
    weights = controller.weights(utility)
    assert torch.equal(weights[0], torch.zeros(3))
    assert torch.allclose(weights[1], torch.tensor([.25, 0., .75]))
    assert (weights >= 0).all() and (weights <= 1).all()
    s = Student(4)
    x = torch.rand(1, 3, 16, 16)
    y, f = s(x, True)
    controller(*controller_inputs(x, y, f)).sum().backward()
    assert all(p.grad is None for p in s.parameters())
    assert torch.equal(strategy_weights('E0', x), torch.zeros(1, 3))


def test_zero_kd_equals_reconstruction_update():
    s, x, target = Student(4), torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    other = copy.deepcopy(s)
    rec = ReconstructionLoss()
    rec(s(x), target).backward()
    pred = other(x)
    (rec(pred, target)+(KnowledgeViews()(pred, target)*torch.zeros(1, 3)).sum()).backward()
    for a, b in zip(s.parameters(), other.parameters()):
        assert torch.allclose(a.grad, b.grad)


def test_virtual_step_preserves_parameters_and_gradient_buffers():
    s = Student(4).train()
    x, target = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    rec = ReconstructionLoss()
    rec(s(x), target).backward()
    before = copy.deepcopy(s.state_dict())
    gradients = [p.grad.clone() for p in s.parameters()]
    labels = estimate_utility(s, x, target, x, target, KnowledgeViews(), rec, eta=.01, virtual=True)
    assert s.training and labels['virtual_gain'].shape == (3,)
    assert all(torch.equal(v, before[k]) for k, v in s.state_dict().items())
    assert all(torch.equal(p.grad, g) for p, g in zip(s.parameters(), gradients))
    assert all(not v.requires_grad and torch.isfinite(v).all() for v in labels.values())
    # Check virtual gain against a real update on an independent copy.
    clone = copy.deepcopy(s).eval()
    loss = KnowledgeViews()(clone(x), target)[:, 0].mean()
    grads = torch.autograd.grad(loss, tuple(clone.utility_parameters().values()))
    with torch.no_grad():
        for p, g in zip(clone.utility_parameters().values(), grads):
            p -= .01*g
        actual = rec(s(x), target)-rec(clone(x), target)
    assert torch.allclose(labels['virtual_gain'][0], actual, atol=1e-7)


def test_split_leakage_and_validation_readonly():
    with pytest.raises(ValueError, match='leakage'):
        validate_splits({'train': [{'id': 'a', 'input': '/a'}], 'val': [{'id': 'b', 'input': '/a'}]})
    with pytest.raises(ValueError, match='leakage'):
        validate_splits({'train': [{'id': 'a', 'input': '/a'}], 'test': [{'id': 'a', 'input': '/b'}]})
    s = Student(4)
    before = copy.deepcopy(s.state_dict())
    validate(s, [{'input': torch.rand(1, 3, 16, 16), 'target': torch.rand(1, 3, 16, 16)}], 'cpu')
    assert all(torch.equal(v, before[k]) for k, v in s.state_dict().items())
    assert all(p.grad is None for p in s.parameters())


def test_identical_metrics():
    x = torch.rand(2, 3, 16, 16)
    m = metrics(x, x)
    assert torch.equal(m['mse'], torch.zeros(2))
    assert torch.allclose(m['ssim'], torch.ones(2))
