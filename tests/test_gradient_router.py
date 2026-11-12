import torch
from luad_uie.models.student import Student
from luad_uie.models.gradient_router import HeadGradientRouter


def test_per_image_head_vectors_match_autograd_and_reject_conflict():
    torch.manual_seed(42)
    model = Student(4)
    pred, features = model(torch.rand(2, 3, 9, 13), True)
    loss = pred.square().mean((1, 2, 3))
    vectors = HeadGradientRouter.head_vectors(loss.sum(), pred, features['spatial'])
    for i, image_loss in enumerate(loss):
        grads = torch.autograd.grad(image_loss, tuple(model.head.parameters()), retain_graph=True)
        torch.testing.assert_close(vectors[i], torch.cat([g.flatten() for g in grads]))
    router = HeadGradientRouter()
    # An opposing direction on the second sample should turn off all views there.
    signs = pred.new_tensor([1, -1])
    views = torch.stack([signs * loss, -loss, -loss], dim=1)
    weights, _ = router(pred, features['spatial'], loss.mean(), views, 0)
    torch.testing.assert_close(weights, pred.new_tensor([[1, 0, 0], [0, 0, 0]]))
    assert not weights.requires_grad
    (loss.mean() + (weights * views).sum(1).mean()).backward()
    assert torch.isfinite(model.head.weight.grad).all()
