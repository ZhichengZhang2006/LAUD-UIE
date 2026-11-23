"""Summarize the seven matched 100-epoch development runs after both suites finish."""
import argparse
import json
import math
from pathlib import Path

import torch

from ..runtime import file_hash
from ..datasets import read_manifest
from .train_backbone_fusion import ARMS, BACKBONE_SHA256
from .train_matched_quality import subset


SUITES = {'standard': ARMS, 'fitted': ('free', 'centered')}


def epochs_for_arm(folder):
    rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in rows] != list(range(1, 101)):
        raise ValueError(f'{folder}: expected 100 consecutive epochs')
    for row in rows:
        if row['arm'] != folder.name or not all(math.isfinite(row['val'][k]) for k in ('psnr', 'ssim')):
            raise ValueError(f'{folder}: invalid validation record')
    return rows


def checked_checkpoints(folder):
    rows = epochs_for_arm(folder)
    results = {}
    for name, record in [('last', rows[-1]), ('best', max(rows, key=lambda r: r['val']['psnr']))]:
        path = folder / (name + '.pt')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['arm'] != folder.name or saved['epoch'] != record['epoch']
                or saved['val'] != record['val'] or saved['backbone_sha256'] != BACKBONE_SHA256
                or saved['config']['epochs'] != 100 or saved['config']['evaluation_resize'] != 256):
            raise ValueError(f'{path}: checkpoint and protocol differ')
        results[name] = dict(epoch=record['epoch'], psnr=record['val']['psnr'],
                             ssim=record['val']['ssim'], sha256=file_hash(path))
    return results


def full_lsui_control(directory, development):
    if (directory / 'COMPLETE').read_text().split() != ['free']:
        raise ValueError('Full-data control requires a completed free arm')
    protocol = json.loads((directory / 'protocol.json').read_text())
    args = protocol['parameters']
    expected = dict(epochs=100, batch=1, seed=42, train_limit=0, val_limit=128,
                    head_fit_images=0, arms=['free'], smoke_test=False)
    if (any(args.get(k) != v for k, v in expected.items()) or protocol['head_fit_ids']
            or protocol['weights_sha256'] != BACKBONE_SHA256):
        raise ValueError('Unexpected full-data training protocol')
    data = {}
    for split, limit in [('train', 0), ('val', 128)]:
        manifest = Path(args[split])
        key = f'{split}_sha256'
        if file_hash(manifest) != protocol[key] or protocol[key] != development[key]:
            raise ValueError(f'{split}: the full-data control changed its source manifest')
        data[split] = [row['id'] for row in subset(read_manifest(manifest), limit, 42)]
        if data[split] != protocol[f'{split}_ids']:
            raise ValueError(f'{split}: the full-data control changed its sample inventory')
    if (len(data['train']) != 2711 or data['train'][:512] != development['train_ids']
            or data['val'] != development['val_ids']):
        raise ValueError('Expected nested LSUI training samples and identical validation order')
    return dict(train_images=len(data['train']), val_images=len(data['val']),
                **checked_checkpoints(directory / 'free'))


def collect(root, full_data=False):
    for suite, arms in SUITES.items():
        flag = root / suite / 'COMPLETE'
        if not flag.is_file() or flag.read_text().split() != list(arms):
            raise ValueError(f'{suite}: formal comparison awaits completion')
    summaries = {}
    common = None
    for suite, arms in SUITES.items():
        directory = root / suite
        protocol = json.loads((directory / 'protocol.json').read_text())
        args = protocol['parameters']
        expected = dict(epochs=100, batch=1, seed=42, train_limit=512, val_limit=128,
                        head_fit_images=64 if suite == 'fitted' else 0, arms=list(arms), smoke_test=False)
        if any(args.get(k) != v for k, v in expected.items()) or protocol['weights_sha256'] != BACKBONE_SHA256:
            raise ValueError(f'{suite}: training protocol differs')
        signature = {k:protocol[k] for k in ('train_ids', 'val_ids', 'train_sha256', 'val_sha256')}
        train, val = signature['train_ids'], signature['val_ids']
        if (len(train) != 512 or len(set(train)) != 512 or len(val) != 128 or len(set(val)) != 128
                or set(train).intersection(val) or protocol['head_fit_ids'] != train[:args['head_fit_images']]):
            raise ValueError(f'{suite}: invalid split or initialization sample')
        if common is not None and signature != common:
            raise ValueError('The two suites used different development data')
        common = signature
        for arm in arms:
            folder = directory / arm
            metrics = checked_checkpoints(folder)
            if arm in ('fixed', 'utility'):
                teacher = (folder / 'teacher.sha256').read_text().strip()
                if teacher != summaries['standard/free']['best']['sha256']:
                    raise ValueError(f'{folder}: wrong distillation teacher')
            summaries[f'{suite}/{arm}'] = metrics
    if full_data:
        summaries['full_lsui_free/free'] = full_lsui_control(root / 'full_lsui_free', common)
    return dict(dataset='LSUI development', epochs=100, seed=42, arms=summaries,
                train_manifest_sha256=common['train_sha256'], val_manifest_sha256=common['val_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--include-full-data', action='store_true')
    args = parser.parse_args()
    report = collect(args.root, full_data=args.include_full_data)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
