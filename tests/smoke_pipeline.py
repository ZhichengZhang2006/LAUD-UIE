"""Synthetic integration checks only; never report these scores as research results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from luad_uie.trainers.engine import train
from luad_uie.scripts.evaluate import evaluate
from luad_uie.models import Student
from luad_uie.scripts.analyze_utility import main as audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    manifests = {}
    for split in ('train', 'val', 'test', 'utility_train', 'utility_eval', 'audit_candidates', 'audit_references'):
        manifest = root/f'{split}.csv'
        with manifest.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'input', 'target'])
            for i in range(2):
                target = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
                degraded = (target.astype(float)*np.array([.55, .8, .9])).astype(np.uint8)
                name = f'{split}_{i}'
                Image.fromarray(degraded).save(root/f'{name}_x.png')
                Image.fromarray(target).save(root/f'{name}_y.png')
                writer.writerow([name, f'{name}_x.png', f'{name}_y.png'])
        manifests[split] = str(manifest)
    cfg = {'seed': 42, 'device': args.device, 'student': {'width': 4},
           'teacher': {'mode': 'oracle'}, 'epochs': 1, 'batch_size': 2, 'crop_size': 16,
           'lr': .001, 'ssim_weight': .2, 'kd_weight': .1,
           'data': {k: v for k, v in manifests.items() if not k.startswith('audit')},
           'utility': {'interval': 1, 'virtual_interval': 1, 'warmup_steps': 0, 'eta': .01, 'threshold': .05}}
    for exp, view in [('E0', None), ('E1', 'color'), ('E1', 'illumination'), ('E1', 'detail'), ('E2', None), ('E3', None), ('E4', None), ('E5', None)]:
        name = exp + ('_'+view if view else '')
        config = {**cfg, 'experiment': exp, 'view': view}
        train(config, root/name)
        evaluate(root/name/'best.pt', manifests['test'], root/(name+'_eval'), args.device,
                 baseline=root/'E0_eval'/'metrics.json' if exp != 'E0' else None)
    # Inference manifest has no GT column: student predictions require only RGB input.
    no_gt = root/'no_gt.csv'
    no_gt.write_text('id,input\ntest_0,test_0_x.png\n')
    result = evaluate(root/'E0'/'best.pt', no_gt, root/'no_gt_eval', args.device, save_images=True)
    assert 'psnr' not in result['summary']
    audit(['--run', str(root/'E4'),
                    '--candidates', manifests['audit_candidates'], '--references', manifests['audit_references'],
                    '--output', str(root/'audit'), '--device', args.device])
    # Synthetic random weights exercise checkpoint loading, not a trained teacher experiment.
    checkpoint = root/'synthetic_teacher.pt'
    torch.save(Student(4).state_dict(), checkpoint)
    config = {**cfg, 'experiment': 'E2', 'teacher': {'mode': 'checkpoint', 'width': 4, 'checkpoint': str(checkpoint)}}
    train(config, root/'E2_checkpoint_smoke')
    print('ALL SYNTHETIC PIPELINE CHECKS PASSED; NOT SCIENTIFIC EXPERIMENTS', flush=True)


if __name__ == '__main__':
    main()
