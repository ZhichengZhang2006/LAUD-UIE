"""Fit a train-only global correction probe and test it on the matched validation IDs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest, validate_splits
from ..losses import metrics
from ..runtime import file_hash
from .score_validation_decomposition import components, load_student


def descriptors(image):
    """Compact, fixed moments of luma and RGB for this diagnostic."""
    y = .2126 * image[:, :1] + .7152 * image[:, 1:2] + .0722 * image[:, 2:3]
    flat = torch.cat((image, y), dim=1).flatten(2)
    return torch.cat((flat.mean(2), flat.std(2, unbiased=False),
                      (flat < .15).float().mean(2), (flat > .8).float().mean(2)), dim=1)


def features(image, prediction):
    return torch.cat((descriptors(image), descriptors(prediction)), dim=1)


@torch.inference_mode()
def run(args):
    protocol = json.loads(args.protocol.read_text())
    paths = {'train':args.train, 'val':args.val}
    for name, path in paths.items():
        if protocol[f'{name}_sha256'] != file_hash(path):
            raise ValueError(f'{name}: manifest differs from matched training')
    data = {name:read_manifest(path) for name, path in paths.items()}
    validate_splits(data)
    ids = {name:protocol[f'{name}_ids'] for name in paths}
    ids['train'] = ids['train'][:args.train_limit]
    network = load_student(args.checkpoint, Path('/home/user/MambaCue'), args.device)
    batches = {}
    for part, rows in data.items():
        by_id = {row['id']:row for row in rows}
        batches[part] = DataLoader(PairedDataset([by_id[i] for i in ids[part]], resize=256), batch_size=4)

    fitting_features, fitting_targets = [], []
    for pair in batches['train']:
        image, target = (pair[key].to(args.device) for key in ('input','target'))
        prediction = network(image)
        fitting_features.append(features(image, prediction).cpu().numpy())
        fitting_targets.append((target-prediction).mean((2, 3)).cpu().numpy())
    X = np.concatenate(fitting_features).astype(np.float64)
    Y = np.concatenate(fitting_targets).astype(np.float64)
    center, scale = X.mean(0), X.std(0).clip(.001)
    X = (X-center)/scale
    design = np.column_stack((X, np.ones(len(X))))
    ridge = np.diag(np.r_[np.full(X.shape[1], .01*len(X)), 0.])
    weights = np.linalg.solve(design.T@design+ridge, design.T@Y)
    rows = []
    for pair in batches['val']:
        image, target = (pair[key].to(args.device) for key in ('input','target'))
        prediction = network(image)
        x = (features(image, prediction).cpu().numpy()-center)/scale
        predicted_shift = np.column_stack((x, np.ones(len(x))))@weights
        constant = torch.as_tensor(predicted_shift, dtype=image.dtype, device=image.device)
        corrected = (prediction + constant[:, :, None, None]).clamp(0, 1)
        target_shift = (target-prediction).mean((2, 3), keepdim=True)
        oracle = (prediction + target_shift).clamp(0, 1)
        scores = {}
        for name, estimate in [('base',prediction), ('fitted',corrected), ('target_mean',oracle)]:
            scores[name] = {**metrics(estimate,target), **components(estimate,target)}
        for i, identifier in enumerate(pair['id']):
            rows.append({'id':identifier, 'results':{name:{key:float(value[i]) for key,value in arm.items()}
                                                     for name,arm in scores.items()}})
    summary = {arm:{metric:float(np.mean([row['results'][arm][metric] for row in rows]))
                    for metric in measures}
               for arm,measures in rows[0]['results'].items()}
    report = {'checkpoint_sha256':file_hash(args.checkpoint), 'validation_ids_sha256':hashlib.sha256(
              '\n'.join(ids['val']).encode()).hexdigest(), 'fit_ids':ids['train'],
              'summary':summary, 'validation_count':len(rows), 'per_image':rows}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ('checkpoint','train','val','protocol','output'):
        p.add_argument(arg,type=Path)
    p.add_argument('--train-limit',type=int,default=384)
    p.add_argument('--device',default='cpu')
    torch.set_num_threads(2)
    run(p.parse_args())
