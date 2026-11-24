import numpy as np
import pytest
import torch

from luad_uie.scripts.compare_fusion_validation import paired_statistics, rgb_error_components


def test_paired_sample_differences_and_lower_is_better():
    indices = np.array([[0, 0], [1, 1], [0, 1]])
    s = paired_statistics([0.1, 0.3], [0.2, 0.2], indices, 'ordinal_violation_rate')
    assert s['mean_difference'] == pytest.approx(0.)
    assert s['interval95'] == pytest.approx([-.095,.095])
    assert s['images'] == 2 and s['improved'] == 1 and s['tied'] == 0


def test_paired_statistics_exact_tie_and_finite_checks():
    indices = np.array([[0, 0], [0, 0]])
    s = paired_statistics([1.,1.], [1.,1.], indices, 'ssim')
    assert s['interval95'] == [0.,0.] and s['improved'] == 0 and s['tied'] == 2
    with pytest.raises(ValueError, match='Invalid per-image'):
        paired_statistics([float('nan'), 1.], [1.,1.], indices, 'psnr')


def test_offset_and_spatial_error_partition_total_mse():
    error = torch.tensor([[[[2., 0.]]], [[[3., -1.]]]])
    values = rgb_error_components(error, torch.zeros_like(error))
    torch.testing.assert_close(values['rgb_offset_mse'], torch.tensor([1., 1.]))
    torch.testing.assert_close(values['rgb_spatial_mse'], torch.tensor([1., 4.]))
    torch.testing.assert_close(sum(values.values()), error.square().mean((1, 2, 3)))
