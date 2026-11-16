"""Serial, fail-closed research queue. No automatic recovery after any failure."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import yaml

ROOT = Path('/home/user/luad_uie_prototype_20261031')
PYTHON = '/home/user/Conda/CDF_FFT/bin/python'


def main():
    stage = Path(__file__).resolve().parent
    os.chdir(stage)
    runs = stage/'runs'
    runs.mkdir(exist_ok=True)
    lock = (runs/'progressive.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
    def write():
        (runs/'progressive_status.json').write_text(json.dumps(status, indent=2))
    def run(args):
        status['command'] = args
        write()
        subprocess.run([PYTHON, '-u']+args, check=True)
    try:
        baseline = str(ROOT/'stages/20261112_baseline/runs/E0_LSUI_test/metrics.json')
        for name in ('RG-full', 'RG-scale', 'RG-learned'):
            status['task'] = name
            run(['-m', 'luad_uie.scripts.safe_train', '--config', 'configs/'+name+'_LSUI.yaml',
                 '--output', 'runs/'+name+'_seed42'])
            run(['-m', 'luad_uie.scripts.evaluate', '--checkpoint', 'runs/'+name+'_seed42/best.pt',
                 '--manifest', str(ROOT/'data/LSUI/test.csv'), '--output', 'runs/'+name+'_test', '--baseline', baseline])
        # Preserve the original teacher and its selection history. Extend to total 250.
        status['task'] = 'teacher250'
        original = ROOT/'runs/LSUI_teacher_seed42'
        teacher = runs/'teacher250'
        teacher.mkdir(exist_ok=False)
        for name in ('last.pt', 'best.pt', 'epochs.jsonl'):
            shutil.copy2(original/name, teacher/name)
        cfg = yaml.safe_load((original/'config.yaml').read_text())
        cfg['epochs'] = 250
        cfg['data'] = {k: str(ROOT/v) if v and not Path(v).is_absolute() else v for k,v in cfg['data'].items()}
        path = stage/'configs/teacher250.yaml'
        path.write_text(yaml.safe_dump(cfg))
        run(['-m', 'luad_uie.scripts.safe_train', '--config', str(path), '--output', str(teacher), '--resume', str(teacher/'last.pt')])
        run(['-m', 'luad_uie.scripts.evaluate', '--checkpoint', str(teacher/'best.pt'),
             '--manifest', str(ROOT/'data/LSUI/test.csv'), '--output', 'runs/teacher250_test'])
        status['status'] = 'completed'
    except BaseException as error:
        status['status'] = 'stopped'
        status['error'] = repr(error)
        status['automatic_retry'] = False
        raise
    finally:
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write()


if __name__ == '__main__':
    main()
