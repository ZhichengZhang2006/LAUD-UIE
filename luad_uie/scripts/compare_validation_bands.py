"""Compare independently scored validation checkpoints on the exact same image set."""

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ('psnr', 'ssim', 'mse', 'global_neutral', 'local_neutral', 'detail_neutral',
           'global_chroma', 'local_chroma', 'detail_chroma')


def compare(reference, candidate):
    for key in ('val_sha256', 'validation_ids_sha256', 'count'):
        if reference[key] != candidate[key]:
            raise ValueError(f'Validation provenance differs: {key}')
    rows = [{row['id']: row for row in report['per_image']} for report in (reference, candidate)]
    if set(rows[0]) != set(rows[1]) or len(rows[0]) != reference['count'] or not rows[0]:
        raise ValueError('Validation predictions do not cover the same unique image IDs')
    ids = sorted(rows[0])
    sampling = np.random.default_rng(42).integers(len(ids), size=(2000, len(ids)))
    bounds = {}
    for metric in METRICS:
        a, b = (np.asarray([source[identifier][metric] for identifier in ids]) for source in rows)
        delta = b - a
        bounds[metric] = {
            'reference': float(a.mean()),
            'candidate': float(b.mean()),
            'candidate_minus_reference': float(delta.mean()),
            'paired_image_interval_95': np.quantile(delta[sampling].mean(1), [.025, .975]).tolist(),
        }
    return {
        'checkpoint_sha256': [report['checkpoint_sha256'] for report in (reference, candidate)],
        'validation_ids_sha256': reference['validation_ids_sha256'],
        'images': len(ids),
        'metrics': bounds,
        'interval_scope': 'paired image bootstrap for one training seed',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('reference', 'candidate', 'output'):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding='utf-8')) for path in (args.reference, args.candidate)]
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(compare(*records), stream, indent=2, allow_nan=False)
