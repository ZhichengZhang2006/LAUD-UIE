"""Run the full frozen acceptance protocol on a single selected checkpoint."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(checkpoint, baseline, output):
    output.mkdir(parents=True)
    quality = output / 'candidate_quality'
    downstream = output / 'downstream'
    comparison = output / 'quality_comparison.json'
    timing = output / 'timing.json'
    manifest = Path('/home/user/MambaCue/supplement_20261112/protocol/LSUI_test.json')
    tasks = (
        ('benchmark_mambacue', [quality, '--student-checkpoint', checkpoint]),
        ('score_saved_lpips', [quality/'LSUI', manifest]),
        ('compare_uie_quality', [baseline, quality, comparison]),
        ('profile_uie_pair', [checkpoint, timing]),
        ('benchmark_duo_sift', [checkpoint, downstream]),
        ('summarize_acceptance', [comparison, downstream, timing, output/'summary.json']),
    )
    for name, args in tasks:
        print(f'Running {name}', flush=True)
        command = [sys.executable, '-u', '-m', f'luad_uie.scripts.{name}', *map(str, args)]
        with (output/f'{name}.log').open('x') as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if status.returncode:
            if 'out of memory' in (output/f'{name}.log').read_text(errors='replace').lower():
                (output/'STOPPED_OOM').write_text(name)
            raise subprocess.CalledProcessError(status.returncode, command)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for field in ('checkpoint', 'baseline', 'output'):
        p.add_argument(field, type=Path)
    a = p.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    run(a.checkpoint.resolve(), a.baseline.resolve(), a.output.resolve())
