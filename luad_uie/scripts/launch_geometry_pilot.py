"""Run the four matched geometry and reconstruction arms on a validation split."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def wait_for_cuda():
    check = [sys.executable, '-c', "import torch; torch.zeros(1,device='cuda')"]
    while True:
        probe = subprocess.run(check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if probe.returncode == 0:
            return
        if 'driver shutting down' not in probe.stdout:
            raise RuntimeError(probe.stdout)
        print('Waiting for CDF_FFT CUDA initialization', flush=True)
        time.sleep(90)


def run(base, output, epochs):
    raw = yaml.safe_load(base.read_text())
    # Two contrasts isolate preprocessing and centered subpixel correction.
    arms = {
        'gated_crop': ({'architecture': 'selective', 'gated': True}, None),
        'gated_resize': ({'architecture': 'selective', 'gated': True}, 256),
        'packed_mean': ({'architecture': 'packed', 'phase_residual': False}, 256),
        'packed_phase': ({'architecture': 'packed', 'phase_residual': True}, 256),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, (options, working_size) in arms.items():
        config = {**raw, 'student': {**options, 'width': 24}, 'working_size': working_size,
                  'validation_working_size': 256, 'epochs': epochs}
        config['data'] = {key: value for key, value in config['data'].items() if key != 'test'}
        config_file = output / f'{name}.yaml'
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    for name in arms:
        print(json.dumps({'arm': name, 'status': 'preflight'}), flush=True)
        wait_for_cuda()
        print(json.dumps({'arm': name, 'status': 'training'}), flush=True)
        subprocess.run([sys.executable, '-u', '-m', 'luad_uie.scripts.safe_train',
                        '--config', str(output / f'{name}.yaml'), '--output', str(output / name)],
                       check=True)
        # Score each arm at the same full-image resize. Re-run only the frozen
        # checkpoint on validation, following the predeclared selection rule.
        cfg = yaml.safe_load((output / f'{name}.yaml').read_text())
        subprocess.run([sys.executable, '-u', '-m', 'luad_uie.scripts.score_geometry_validation',
                        str(output / name), cfg['data']['val']], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('base', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--epochs', type=int, default=20)
    args = p.parse_args()
    run(args.base, args.output, args.epochs)
