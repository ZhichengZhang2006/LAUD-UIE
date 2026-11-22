"""Evaluate synthetic particle removal and edge preservation in disjoint test regions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmark_uie_downstream import models, digest
from luad_uie.losses.particles import add_particles, gradient_pairs


def read_image(path):
    with Image.open(path) as source:
        rgb = np.array(source.convert('RGB').resize((256, 256), Image.BILINEAR), dtype=np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda() / 255


def score_regions(estimated, corrected, target, mask):
    selected = {'particle_non_edge': [], 'particle_edge': [], 'unperturbed_edge': []}
    axes = (mask[..., :-1] | mask[..., 1:], mask[..., :-1, :] | mask[..., 1:, :])
    for base, pred, clean, hit in zip(gradient_pairs(estimated), gradient_pairs(corrected),
                                      gradient_pairs(target), axes):
        true_edge = clean.abs() >= .02
        error = (pred-clean).abs()
        increase = error - (base-clean).abs()
        for name, region in [('particle_non_edge', hit & ~true_edge),
                             ('particle_edge', hit & true_edge),
                             ('unperturbed_edge', ~hit & true_edge)]:
            selected[name].append((error[region], increase[region]))
    result = {}
    for name, axes_scores in selected.items():
        error, increase = (torch.cat([axis[i] for axis in axes_scores]) for i in range(2))
        result[name] = {'pixels': int(error.numel()), 'error': float(error.mean()) if error.numel() else None,
                        'change_due_to_particles': float(increase.mean()) if error.numel() else None}
    return result


@torch.inference_mode()
def run(checkpoint, manifest, output):
    networks, baseline = models(checkpoint)
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {name: [] for name in networks}
    for row in rows:
        x = read_image(manifest.parent/row['input'])
        target = read_image(manifest.parent/row['target'])
        seed = int(digest_for_id(row['id'])[:8], 16) + 20261122
        changed, mask = add_particles(x, seed)
        for name, model in networks.items():
            region = score_regions(model(x), model(changed), target, mask)
            result[name].append({'id':row['id'], 'regions':region})
    with output.open('x') as stream:
        json.dump({'manifest_sha256':digest(manifest), 'candidate':digest(checkpoint),
                   'mambacue':baseline, 'per_image':result}, stream, allow_nan=False)


def digest_for_id(image_id):
    from hashlib import sha256
    return sha256(image_id.encode()).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ('checkpoint', 'manifest', 'output'):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    run(args.checkpoint, args.manifest, args.output)
