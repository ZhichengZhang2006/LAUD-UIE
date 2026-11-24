"""Paired LSUI validation differences from the full-budget final checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest
from ..losses import metrics
from ..losses.ordinal_structure import structure_metrics
from ..models.selective_student import build_student
from ..runtime import file_hash, seed_all
from .summarize_fusion_100 import checked_checkpoints, collect
from .train_backbone_fusion import ARMS, BACKBONE_SHA256
from .train_matched_quality import subset


CONTRASTS = (
    ('centered', 'standard/centered', 'standard/free'),
    ('ordinal', 'standard/ordinal', 'standard/centered'),
    ('fixed_distillation', 'standard/fixed', 'standard/ordinal'),
    ('utility_vs_fixed', 'standard/utility', 'standard/fixed'),
    ('utility_vs_no_distillation', 'standard/utility', 'standard/ordinal'),
    ('centered_with_fitted_head', 'fitted/centered', 'fitted/free'),
    ('fitted_free_head', 'fitted/free', 'standard/free'),
    ('fitted_centered_head', 'fitted/centered', 'standard/centered'),
)
METRICS = ('psnr', 'ssim', 'ordinal_violation_rate', 'rgb_offset_mse', 'rgb_spatial_mse')
SIFT_METRICS = ('sift_keypoints', 'sift_matched_points')
LOWER_IS_BETTER = {'ordinal_violation_rate', 'rgb_offset_mse', 'rgb_spatial_mse'}


def rgb_error_components(prediction, target):
    """Partition per-image MSE into global offset and centered spatial error."""
    error = prediction - target
    offset = error.mean(dim=(-2, -1), keepdim=True)
    return {'rgb_offset_mse':offset.square().mean(dim=(1, 2, 3)),
            'rgb_spatial_mse':(error-offset).square().mean(dim=(1, 2, 3))}


def paired_statistics(candidate, baseline, indices, metric):
    differences = np.asarray(candidate) - np.asarray(baseline)
    if differences.ndim != 1 or not np.isfinite(differences).all():
        raise ValueError(f'Invalid per-image values for {metric}')
    sampling = differences[indices].mean(axis=1)
    favorable = differences < 0 if metric in LOWER_IS_BETTER else differences > 0
    return dict(mean_difference=float(differences.mean()),
                median_difference=float(np.median(differences)),
                interval95=np.quantile(sampling, [.025, .975]).tolist(),
                improved=int(favorable.sum()), tied=int((differences == 0).sum()),
                images=len(differences))


@torch.inference_mode()
def score_arm(checkpoint, expected, dataset, with_sift=False):
    if file_hash(checkpoint) != expected['sha256']:
        raise ValueError(f'Changed checkpoint: {checkpoint}')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = build_student(saved['config']['student'])
    model.load_state_dict(saved['student'])
    model.cuda().eval()
    if with_sift:
        import cv2
        from .benchmark_duo_sift import SIZE, SIFT_TRANSFORMS, repeatability, sift_points, tensor_rgb

        cv2.setNumThreads(1)
        sift = cv2.SIFT_create()
        transforms = [cv2.getRotationMatrix2D(((SIZE-1)/2, (SIZE-1)/2), angle, scale)
                      for angle, scale in SIFT_TRANSFORMS]
    measurements = []
    for batch in DataLoader(dataset, batch_size=1, num_workers=2):
        x, y = batch['input'].cuda(), batch['target'].cuda()
        prediction = model(x)
        values = metrics(prediction, y)
        values.update(structure_metrics(prediction, y, x))
        values.update(rgb_error_components(prediction, y))
        row = dict(id=batch['id'][0], **{key:values[key].item() for key in METRICS})
        if with_sift:
            rgb = batch['input'][0].permute(1, 2, 0).mul(255).round().to(torch.uint8).contiguous().numpy()
            count, points = sift_points(sift, prediction)
            matches = []
            for affine in transforms:
                warped = cv2.warpAffine(rgb, affine, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
                _, warped_points = sift_points(sift, model(tensor_rgb(warped)))
                matches.append(repeatability(points, warped_points, affine))
            row.update(sift_keypoints=count, sift_matched_points=sum(m['matches'] for m in matches),
                       sift_transforms=matches)
        measurements.append(row)
    for key in ('psnr', 'ssim'):
        if abs(np.mean([row[key] for row in measurements]) - expected[key]) > .001:
            raise ValueError(f'{checkpoint}: final validation {key} does not reproduce its log')
    return measurements


def verify_gradient_control(directory, standard):
    if (directory / 'COMPLETE').read_text().split() != ['gradient']:
        raise ValueError('The gradient control must finish before comparison')
    protocol = json.loads((directory / 'protocol.json').read_text())
    for key in ('weights_sha256', 'train_sha256', 'val_sha256', 'train_ids', 'val_ids', 'head_fit_ids'):
        if protocol[key] != standard[key]:
            raise ValueError(f'Gradient control differs in {key}')
    for key in ('batch', 'epochs', 'seed', 'head_fit_images', 'train_limit', 'val_limit', 'smoke_test'):
        if protocol['parameters'][key] != standard['parameters'][key]:
            raise ValueError(f'Gradient control differs in {key}')
    return checked_checkpoints(directory / 'gradient')


def run(root, output, gradient_control=None, with_sift=False, centered_only=False):
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((root / 'standard/protocol.json').read_text())
    if centered_only:
        args = protocol['parameters']
        expected = dict(epochs=100, batch=1, seed=42, train_limit=512, val_limit=128,
                        head_fit_images=0, arms=list(ARMS), smoke_test=False)
        if (any(args.get(k) != v for k, v in expected.items()) or protocol['head_fit_ids']
                or protocol['weights_sha256'] != BACKBONE_SHA256):
            raise ValueError('The centered comparison requires the frozen standard protocol')
        for split, limit in [('train', 512), ('val', 128)]:
            split_path = Path(args[split])
            ordered = subset(read_manifest(split_path), limit, 42)
            if (file_hash(split_path) != protocol[f'{split}_sha256']
                    or len(ordered) != limit
                    or [row['id'] for row in ordered] != protocol[f'{split}_ids']):
                raise ValueError(f'The original {split} manifest or sample order changed')
        arms = {f'standard/{arm}':checked_checkpoints(root / 'standard' / arm)
                for arm in ('free', 'centered')}
        verified = dict(arms=arms, val_manifest_sha256=protocol['val_sha256'])
        specifications = CONTRASTS[:1]
    else:
        verified = collect(root)
        specifications = CONTRASTS
    manifest = Path(protocol['parameters']['val'])
    if file_hash(manifest) != verified['val_manifest_sha256']:
        raise ValueError('The original validation manifest changed')
    lookup = {row['id']:row for row in read_manifest(manifest)}
    ids = protocol['val_ids']
    dataset = PairedDataset([lookup[i] for i in ids], resize=256)
    seed_all(42)
    selected = {arm:(root / arm / 'last.pt', evidence['last']) for arm, evidence in verified['arms'].items()}
    if gradient_control is not None:
        control = verify_gradient_control(gradient_control, protocol)
        selected['control/gradient'] = gradient_control / 'gradient/last.pt', control['last']
        specifications += (('gradient', 'control/gradient', 'standard/centered'),
                           ('ordinal_vs_gradient', 'standard/ordinal', 'control/gradient'))
    measurements = {}
    for arm, (checkpoint, evidence) in selected.items():
        values = score_arm(checkpoint, evidence, dataset, with_sift)
        if [row['id'] for row in values] != ids:
            raise ValueError(f'{arm}: validation order changed')
        measurements[arm] = values
    indices = np.random.default_rng(42).integers(0, len(ids), size=(5000, len(ids)))
    contrasts = {}
    for name, candidate, baseline in specifications:
        contrasts[name] = dict(candidate=candidate, baseline=baseline, metrics={
            metric:paired_statistics([row[metric] for row in measurements[candidate]],
                                     [row[metric] for row in measurements[baseline]], indices, metric)
            for metric in METRICS + (SIFT_METRICS if with_sift else ())})
    kd_weights = {}
    if not centered_only:
        for arm in ('fixed', 'utility'):
            history = (root / 'standard' / arm / 'epochs.jsonl').read_text().splitlines()
            kd_weights[arm] = [json.loads(line)['kd_weights'] for line in history]
    scope = 'centered_vs_free' if centered_only else 'full_ablation'
    report = dict(dataset='LSUI validation', scope=scope, checkpoint_epoch=100, seed=42, bootstrap_draws=5000,
                  val_manifest_sha256=verified['val_manifest_sha256'],
                  checkpoints={key:value[1]['sha256'] for key, value in selected.items()},
                  contrasts=contrasts, per_image=measurements, training_kd_weights_by_epoch=kd_weights)
    if with_sift:
        from .benchmark_duo_sift import BORDER, SIFT_TRANSFORMS, TOLERANCE
        import cv2

        report['sift_protocol'] = dict(opencv=cv2.__version__, transforms=SIFT_TRANSFORMS,
                                       border_px=BORDER, match_tolerance_px=TOLERANCE)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--gradient-control', type=Path)
    scope.add_argument('--centered-only', action='store_true')
    parser.add_argument('--with-sift', action='store_true')
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.gradient_control, args.with_sift, args.centered_only)
