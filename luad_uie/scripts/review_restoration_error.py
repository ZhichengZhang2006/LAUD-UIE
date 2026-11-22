"""Decompose frozen restoration errors into orthogonal spatial and RGB components."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squared_error_terms(prediction, reference, cell):
    error = np.asarray(prediction, dtype=np.float64) - reference
    height, width, bands = error.shape
    if bands != 3 or height % cell or width % cell:
        raise ValueError('Cell side must divide both image dimensions')
    blocks = error.reshape(height//cell, cell, width//cell, cell, 3)
    block_mean = blocks.mean(axis=(1, 3), keepdims=True)
    components = {}
    for frequency, part in [('smooth', block_mean), ('detail', blocks-block_mean)]:
        neutral = part.mean(axis=-1, keepdims=True)
        components[f'{frequency}_neutral'] = float(np.mean(neutral**2))
        components[f'{frequency}_opponent'] = float(np.mean((part-neutral)**2))
    mse = float(np.mean(error**2))
    if not np.isclose(mse, sum(components.values()), rtol=1e-9, atol=1e-14):
        raise ValueError('Error projection did not preserve total energy')
    return {'mse':mse, **components}


def read_reference(record):
    path = Path(record['target'])
    if digest(path) != record['target_sha256']:
        raise ValueError(f'Reference changed: {path}')
    with Image.open(path) as im:
        im = im.convert('RGB').resize((256, 256), Image.BILINEAR)
        return np.asarray(im, dtype=np.float64) / 255


def analyze(manifest, runs, cell):
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    for run in runs.values():
        provenance = json.loads((run.parent/'provenance.json').read_text())
        if provenance['manifest_sha256'] != digest(manifest):
            raise ValueError('Manifests differ')
        indexed_ids = [json.loads(line)['id'] for line in (run/'metrics.jsonl').read_text().splitlines()]
        if indexed_ids != [row['name'] for row in rows]:
            raise ValueError('Prediction array order differs from reference IDs')
    measurements = []
    for idx, row in enumerate(rows):
        reference = read_reference(row)
        results = {name: squared_error_terms(np.load(run/f'{idx}.npy', allow_pickle=False), reference, cell)
                   for name, run in runs.items()}
        measurements.append({'id':row['name'], 'metrics':results})
    keys = measurements[0]['metrics']['candidate']
    summary = {name:{key:float(np.mean([row['metrics'][name][key] for row in measurements]))
                     for key in keys} for name in runs}
    excess = {key:summary['candidate'][key]-summary['mambacue'][key] for key in keys}
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(rows), size=(2000,len(rows)))
    ci = {}
    for key in keys:
        delta = np.array([row['metrics']['candidate'][key]-row['metrics']['mambacue'][key]
                          for row in measurements])
        ci[key] = np.quantile(delta[indices].mean(axis=1), [.025,.975]).tolist()
    return {'manifest_sha256':digest(manifest), 'images':len(rows), 'cell_side':cell, 'summary':summary,
            'candidate_excess':excess, 'paired_image_interval_95':ci, 'per_image':measurements}


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    for arg in ('manifest','baseline','candidate','output'):
        cli.add_argument(arg, type=Path)
    cli.add_argument('--cell', type=int, default=16)
    args = cli.parse_args()
    report = analyze(args.manifest, {'mambacue':args.baseline, 'candidate':args.candidate}, args.cell)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='per_image'},indent=2),flush=True)
