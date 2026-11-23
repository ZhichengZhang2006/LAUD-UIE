"""Summarize the seven matched 100-epoch development runs after both suites finish."""
import argparse
import json
import math
from pathlib import Path

import torch

from ..runtime import file_hash
from .train_backbone_fusion import ARMS, BACKBONE_SHA256


SUITES = {'standard': ARMS, 'fitted': ('free', 'centered')}


def epochs_for_arm(folder):
    rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in rows] != list(range(1, 101)):
        raise ValueError(f'{folder}: expected 100 consecutive epochs')
    for row in rows:
        if row['arm'] != folder.name or not all(math.isfinite(row['val'][k]) for k in ('psnr', 'ssim')):
            raise ValueError(f'{folder}: invalid validation record')
    return rows


def collect(root):
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
            rows = epochs_for_arm(folder)
            metrics = {}
            for name, record in [('last', rows[-1]), ('best', max(rows, key=lambda r: r['val']['psnr']))]:
                path = folder / (name + '.pt')
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if (saved['arm'] != arm or saved['epoch'] != record['epoch']
                        or saved['val'] != record['val'] or saved['backbone_sha256'] != BACKBONE_SHA256
                        or saved['config']['epochs'] != 100 or saved['config']['evaluation_resize'] != 256):
                    raise ValueError(f'{path}: checkpoint and protocol differ')
                metrics[name] = dict(epoch=record['epoch'], psnr=record['val']['psnr'],
                                     ssim=record['val']['ssim'], sha256=file_hash(path))
            if arm in ('fixed', 'utility'):
                teacher = (folder / 'teacher.sha256').read_text().strip()
                if teacher != summaries['standard/free']['best']['sha256']:
                    raise ValueError(f'{folder}: wrong distillation teacher')
            summaries[f'{suite}/{arm}'] = metrics
    return dict(dataset='LSUI development', epochs=100, seed=42, arms=summaries,
                train_manifest_sha256=common['train_sha256'], val_manifest_sha256=common['val_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = collect(args.root)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
