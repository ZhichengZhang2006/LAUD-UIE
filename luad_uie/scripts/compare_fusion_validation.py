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
from .summarize_fusion_100 import collect


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
METRICS = ('psnr', 'ssim', 'ordinal_violation_rate')
LOWER_IS_BETTER = {'ordinal_violation_rate'}


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
def score_arm(checkpoint, expected, dataset):
    if file_hash(checkpoint) != expected['sha256']:
        raise ValueError(f'Changed checkpoint: {checkpoint}')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = build_student(saved['config']['student'])
    model.load_state_dict(saved['student'])
    model.cuda().eval()
    measurements = []
    for batch in DataLoader(dataset, batch_size=1, num_workers=2):
        x, y = batch['input'].cuda(), batch['target'].cuda()
        prediction = model(x)
        values = metrics(prediction, y)
        values.update(structure_metrics(prediction, y, x))
        measurements.append(dict(id=batch['id'][0], **{key:values[key].item() for key in METRICS}))
    for key in ('psnr', 'ssim'):
        if abs(np.mean([row[key] for row in measurements]) - expected[key]) > .001:
            raise ValueError(f'{checkpoint}: final validation {key} does not reproduce its log')
    return measurements


def run(root, output):
    verified = collect(root)
    protocol = json.loads((root / 'standard/protocol.json').read_text())
    manifest = Path(protocol['parameters']['val'])
    if file_hash(manifest) != verified['val_manifest_sha256']:
        raise ValueError('The original validation manifest changed')
    lookup = {row['id']:row for row in read_manifest(manifest)}
    ids = protocol['val_ids']
    dataset = PairedDataset([lookup[i] for i in ids], resize=256)
    seed_all(42)
    measurements = {}
    for arm, evidence in verified['arms'].items():
        values = score_arm(root / arm / 'last.pt', evidence['last'], dataset)
        if [row['id'] for row in values] != ids:
            raise ValueError(f'{arm}: validation order changed')
        measurements[arm] = values
    indices = np.random.default_rng(42).integers(0, len(ids), size=(5000, len(ids)))
    contrasts = {}
    for name, candidate, baseline in CONTRASTS:
        contrasts[name] = dict(candidate=candidate, baseline=baseline, metrics={
            metric:paired_statistics([row[metric] for row in measurements[candidate]],
                                     [row[metric] for row in measurements[baseline]], indices, metric)
            for metric in METRICS})
    report = dict(dataset='LSUI validation', checkpoint_epoch=100, seed=42, bootstrap_draws=5000,
                  val_manifest_sha256=verified['val_manifest_sha256'],
                  checkpoints={key:value['last']['sha256'] for key, value in verified['arms'].items()},
                  contrasts=contrasts, per_image=measurements)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
