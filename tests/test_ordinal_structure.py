import torch

from luad_uie.losses.ordinal_structure import full_gradient, ordinal_structure, structure_metrics


def test_consensus_keeps_order_without_copying_contrast():
    ramp = torch.linspace(.1, .8, 16).repeat(1, 3, 12, 1)
    softer = .1 + ramp * .65
    reference = .05 + ramp * .8
    assert ordinal_structure(softer, ramp, reference).item() == 0
    assert full_gradient(softer, reference).item() > 0
    reversed_order = (1 - ramp).requires_grad_()
    penalty = ordinal_structure(reversed_order, ramp, reference)
    penalty.backward()
    assert penalty.item() > 0
    assert reversed_order.grad.abs().sum() > 0
    score = structure_metrics(reversed_order, reference, ramp)
    assert score['ordinal_violation_rate'].item() == 1
    assert score['ordinal_anchor_fraction'].item() > 0


def test_flat_and_disagreeing_pairs_supply_no_ordinal_target():
    ramp = torch.linspace(.1, .8, 16).repeat(1, 3, 12, 1)
    assert ordinal_structure(ramp, ramp, 1-ramp).item() == 0
    flat = torch.full_like(ramp, .5)
    assert ordinal_structure(1-ramp, flat, ramp).item() == 0
    assert structure_metrics(ramp, 1-ramp, ramp)['ordinal_anchor_fraction'].item() == 0
