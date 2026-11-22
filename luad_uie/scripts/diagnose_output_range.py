"""Validation-only diagnostic: fit output controls per image and estimate expressivity."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
from torch.nn import functional as F

from ..datasets import PairedDataset, read_manifest
from ..losses import metrics
from ..models.selective_student import local_mean


def render(x, control, packed):
    if packed:
        correction = control.tanh().unflatten(1, (3, 4))
        mean = correction.mean(dim=2)
        smooth = F.interpolate(mean, scale_factor=2, mode='bilinear', align_corners=False)
        return (x + smooth + F.pixel_shuffle((correction - mean[:, :, None]).flatten(1, 2), 2)).clamp(0, 1)
    resized = F.interpolate(control, scale_factor=2, mode='bilinear', align_corners=False)
    light, red, blue, detail = resized.tanh().split(1, dim=1)
    green = -(.2126 * red + .0722 * blue) / .7152
    luminance = .2126*x[:, :1] + .7152*x[:, 1:2] + .0722*x[:, 2:3]
    high = detail * (luminance - local_mean(luminance))
    return (x + local_mean(light) + torch.cat((red, green, blue), 1) + high - local_mean(high)).clamp(0, 1)


def fit(image, target, packed, steps):
    width = 12 if packed else 4
    controls = torch.zeros((1, width, 128, 128), device=image.device, requires_grad=True)
    optimizer = torch.optim.Adam((controls,), lr=.035)
    for _ in range(steps):
        estimate = render(image, controls, packed)
        loss = (estimate-target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return render(image, controls, packed)


def run(manifest, checkpoint_file, output, limit, steps):
    repo = Path('/home/user/MambaCue')
    sys.path.insert(0, str(repo))
    from model import myModel
    from ..models.selective_student import build_student

    baseline = json.loads((repo/'supplement_20261112/protocol/historical_checkpoints.json').read_text())['LSUI']
    if hashlib.sha256(Path(baseline['path']).read_bytes()).hexdigest() != baseline['sha256']:
        raise ValueError('MambaCue checkpoint mismatch')
    saved = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
    if Path(saved['config']['data']['val']).resolve() != manifest.resolve():
        raise ValueError('Diagnostic accepts validation data only')
    student = build_student(saved['config']['student']).eval().cuda()
    student.load_state_dict(saved['student'])
    mamba = myModel().eval().cuda()
    mamba.load_state_dict(torch.load(baseline['path'], map_location='cpu', weights_only=True))
    rows = sorted(read_manifest(manifest), key=lambda row: hashlib.sha256(f"42:{row['id']}".encode()).hexdigest())
    data = PairedDataset(rows[:limit], resize=256)
    outcomes = []
    for sample in data:
        x, y = (sample[name][None].cuda() for name in ('input', 'target'))
        with torch.no_grad():
            predictions = {'gated': student(x), 'mambacue': mamba(x)}
        predictions['gain_oracle'] = fit(x, y, False, steps)
        predictions['phase_oracle'] = fit(x, y, True, steps)
        estimates = {name: {m:float(v.item()) for m, v in metrics(pred, y).items()}
                     for name, pred in predictions.items()}
        outcomes.append({'id':sample['id'], 'metrics':estimates})
    summary = {name:{key:statistics.mean(row['metrics'][name][key] for row in outcomes)
                     for key in ('psnr','ssim','mse')}
               for name in outcomes[0]['metrics']}
    with output.open('x') as f:
        json.dump({'steps':steps, 'images':len(outcomes), 'oracle':'uses each validation target to fit controls',
                   'summary':summary, 'per_image':outcomes}, f, indent=2, allow_nan=False)
    print(summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('manifest','checkpoint','output'):
        p.add_argument(name, type=Path)
    p.add_argument('--limit', type=int, default=32)
    p.add_argument('--steps', type=int, default=60)
    opts = p.parse_args()
    run(opts.manifest, opts.checkpoint, opts.output, opts.limit, opts.steps)
