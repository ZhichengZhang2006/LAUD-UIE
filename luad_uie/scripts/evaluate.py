import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from ..models import Student
from ..datasets import read_manifest, PairedDataset
from ..losses import metrics
from ..runtime import file_hash


def validate_baseline(base, config, manifest_hash):
    required = ('student', 'seed', 'epochs', 'crop_size', 'batch_size', 'lr')
    reference = base.get('training_config')
    if reference is None:
        raise ValueError('Legacy baseline lacks training_config; re-evaluate the actual student E0')
    if base['experiment'] != 'E0' or base['summary']['manifest_sha256'] != manifest_hash:
        raise ValueError('Baseline must be E0 on the same test manifest')
    for key in required:
        if reference.get(key) != config.get(key):
            raise ValueError(f'Baseline mismatch: {key}')
    if reference.get('ssim_weight', .2) != config.get('ssim_weight', .2):
        raise ValueError('Baseline reconstruction objective mismatch')
    expected = base.get('training_manifest_sha256', {})
    for split in ('train', 'val'):
        path = config.get('data', {}).get(split)
        if not path or expected.get(split) != file_hash(path):
            raise ValueError(f'Baseline mismatch: {split} manifest')


@torch.no_grad()
def evaluate(checkpoint, manifest, output, device='cuda', baseline=None, save_images=False, inference_only=False):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = state['config']
    manifest_hash = file_hash(manifest)
    base = json.loads(Path(baseline).read_text(encoding='utf-8')) if baseline else None
    if base is not None:
        validate_baseline(base, config, manifest_hash)
    student = Student(state['config']['student']['width']).to(device).eval()
    student.load_state_dict(state['student'])
    records = read_manifest(manifest, require_target=False)
    if inference_only:
        records = [{**r, 'target': None} for r in records]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for batch in DataLoader(PairedDataset(records), batch_size=1):
        x = batch['input'].to(device)
        # Warm up per shape; timing excludes IO, transfer and metrics, batch=1.
        for _ in range(3):
            student(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = student(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        row = {'id': batch['id'][0], 'latency_ms': (time.perf_counter()-started)*1000,
               'height': x.shape[-2], 'width': x.shape[-1]}
        if 'target' in batch:
            row.update({k: v.item() for k, v in metrics(prediction, batch['target'].to(device)).items()})
        if save_images:
            rgb = (prediction[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(rgb).save(out/f'{len(rows):06d}.png')
        rows.append(row)
    summary = {key: float(np.mean([r[key] for r in rows if key in r])) for key in ('psnr', 'ssim', 'mse', 'latency_ms') if any(key in r for r in rows)}
    summary.update({'images': len(rows), 'teacher_mode_during_training': state['config']['teacher']['mode'],
                    'manifest_sha256': file_hash(manifest), 'student_only': True})
    if base is not None:
        by_id = {r['id']: r for r in base['per_image']}
        if set(by_id) != {r['id'] for r in rows} or any('psnr' not in r or 'psnr' not in by_id[r['id']] for r in rows):
            raise ValueError('Baseline comparison requires matching IDs and GT metrics for every image')
        for row in rows:
            row['delta_psnr_vs_E0'] = row['psnr']-by_id[row['id']]['psnr']
        summary['degraded_fraction_vs_E0'] = sum(r['delta_psnr_vs_E0'] < 0 for r in rows)/len(rows)
    result = {'experiment': config['experiment'], 'summary': summary, 'per_image': rows,
              'training_config': config, 'checkpoint_sha256': file_hash(checkpoint),
              'training_manifest_sha256': {k: file_hash(v) for k, v in config.get('data', {}).items() if v}}
    (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--baseline', help='E0 metrics.json')
    p.add_argument('--save-images', action='store_true')
    p.add_argument('--inference-only', action='store_true', help='Do not load GT pixels or compute GT metrics')
    args = p.parse_args()
    evaluate(**vars(args))


if __name__ == '__main__':
    main()
