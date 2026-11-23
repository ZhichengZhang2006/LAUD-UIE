"""Run final acceptance with frozen LSUI and UIEB checkpoints."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(checkpoint, baseline, output, uieb_candidate, uieb_audit):
    output.mkdir(parents=True)
    quality = output / 'candidate_quality'
    uieb_base = output / 'uieb_baseline'
    uieb_quality = output / 'uieb_candidate'
    downstream = output / 'downstream'
    comparison = output / 'quality_comparison.json'
    timing = output / 'timing.json'
    manifest = Path('/home/user/MambaCue/supplement_20261112/protocol/LSUI_test.json')
    tasks = (
        ('benchmark_mambacue', [quality, '--student-checkpoint', checkpoint]),
        ('score_saved_lpips', [quality/'LSUI', manifest]),
        ('benchmark_mambacue', [uieb_base, '--uieb-audit', uieb_audit]),
        ('benchmark_mambacue', [uieb_quality, '--uieb-audit', uieb_audit, '--student-checkpoint', uieb_candidate]),
        ('score_saved_lpips', [uieb_base/'UIEB', manifest.with_name('UIEB_test.json')]),
        ('score_saved_lpips', [uieb_quality/'UIEB', manifest.with_name('UIEB_test.json')]),
        ('compare_uie_quality', [baseline, quality, comparison, '--unpaired-baseline', uieb_base,
                                 '--unpaired-candidate', uieb_quality]),
        ('profile_uie_pair', [checkpoint, timing]),
        ('benchmark_duo_sift', [checkpoint, downstream]),
        ('summarize_acceptance', [comparison, downstream, timing, output/'summary.json']),
    )
    for index, (name, args) in enumerate(tasks, 1):
        print(f'Running {name}', flush=True)
        command = [sys.executable, '-u', '-m', f'luad_uie.scripts.{name}', *map(str, args)]
        log_path = output/f'{index:02d}_{name}.log'
        with log_path.open('x') as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if status.returncode:
            if 'out of memory' in log_path.read_text(errors='replace').lower():
                (output/'STOPPED_OOM').write_text(name)
            raise subprocess.CalledProcessError(status.returncode, command)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for field in ('checkpoint', 'baseline', 'output'):
        p.add_argument(field, type=Path)
    p.add_argument('--uieb-candidate', type=Path, required=True)
    p.add_argument('--uieb-audit', type=Path, required=True)
    a = p.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    run(a.checkpoint.resolve(), a.baseline.resolve(), a.output.resolve(),
        a.uieb_candidate.resolve(), a.uieb_audit.resolve())
