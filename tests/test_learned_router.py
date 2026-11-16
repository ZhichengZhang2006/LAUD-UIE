import torch
from luad_uie.models.learned_router import LearnedRouter


def test_predictor_isolation_checkpoint_and_conflict_gate():
    torch.manual_seed(42)
    rng = torch.random.get_rng_state().clone()
    router = LearnedRouter(4, interval=2, warmup_steps=1)
    assert torch.equal(rng, torch.random.get_rng_state())
    degradation = torch.rand(2, 10, requires_grad=True)
    state = torch.rand(2, 14, requires_grad=True)
    prediction = router.predict(degradation, state)
    prediction.square().mean().backward()
    assert degradation.grad is None and state.grad is None
    assert any(p.grad is not None for p in router.predictor.parameters())
    router.teacher_router.ema_norm.copy_(torch.tensor([1., 2., 3.]))
    weights = router.weights(torch.tensor([[1., -1., 0.], [-1., -1., -1.]]), 1)
    torch.testing.assert_close(weights, torch.tensor([[1., 0., 0.], [0., 0., 0.]]))
    warmup = router.weights(torch.full((2, 3), -1.), 0)
    torch.testing.assert_close(warmup.sum(1), torch.ones(2))
    restored = LearnedRouter(4, interval=2, warmup_steps=1)
    restored.load_state_dict(router.state_dict())
    torch.testing.assert_close(restored.predict(degradation, state), prediction)
    torch.testing.assert_close(restored.weights(prediction, 1), router.weights(prediction, 1))
