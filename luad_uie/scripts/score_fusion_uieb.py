"""Score one completed UIEB fusion arm against the frozen workbook entries."""
import argparse
import csv
import json
from math import isfinite
from pathlib import Path

from ..runtime import file_hash


def verify_checkpoint(run, arm, audit_dir):
    import torch

    if arm not in (run / 'COMPLETE').read_text().split():
        raise ValueError('Requested arm is incomplete')
    protocol = json.loads((run / 'protocol.json').read_text())
    parameters = protocol['parameters']
    budget = parameters['epochs']
    history = [json.loads(line) for line in (run / arm / 'epochs.jsonl').read_text().splitlines()]
    if (budget < 100 or parameters.get('smoke_test')
            or [row['epoch'] for row in history] != list(range(1, budget + 1))):
        raise ValueError('Formal scoring requires the full declared budget of at least 100 epochs')
    if any(row['arm'] != arm or not all(isfinite(row['val'][metric]) for metric in ('psnr', 'ssim'))
           for row in history):
        raise ValueError('Unexpected arm or non-finite validation score in UIEB history')
    audit = json.loads((audit_dir / 'initialization.json').read_text())
    checkpoint = run / arm / 'best.pt'
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if (state['arm'] != arm or state['config']['student']['architecture'] != 'backbone_fusion'
            or state['backbone_sha256'] != audit['backbone_sha256']
            or protocol['weights_sha256'] != state['backbone_sha256']):
        raise ValueError('Unexpected model or pretraining source')
    best = max(history, key=lambda row: row['val']['psnr'])
    if state['epoch'] != best['epoch'] or state['val'] != best['val']:
        raise ValueError('The selected UIEB checkpoint is not the recorded best validation epoch')
    last = torch.load(run / arm / 'last.pt', weights_only=True, map_location='cpu')
    if (last['arm'] != arm or last['epoch'] != budget or last['val'] != history[-1]['val']
            or last['config'] != state['config'] or last['backbone_sha256'] != state['backbone_sha256']
            or state['config']['epochs'] != budget or state['config']['evaluation_resize'] != 256
            or state['config']['seed'] != parameters['seed']
            or state['config']['batch_size'] != parameters['batch']):
        raise ValueError('UIEB final checkpoint disagrees with the declared training protocol')
    for split in ('train', 'val'):
        manifest = Path(state['config']['data'][split])
        digest = file_hash(manifest)
        if digest != audit['candidate_manifest_sha256'][split] or digest != protocol[f'{split}_sha256']:
            raise ValueError(f'{split} source differs from the UIEB initialization audit')
        with manifest.open(newline='', encoding='utf-8-sig') as stream:
            ids = [r['id'] for r in csv.DictReader(stream)]
        used = protocol[f'{split}_ids']
        if len(ids) != len(used) or set(used) != set(ids):
            raise ValueError(f'Formal evaluation requires the full {split} split')
    return checkpoint, state['epoch']


def workbook_comparison(summaries, workbook):
    names = {'CH60': 'C 60', 'U45': 'U45', 'UIEB': 'UIEB'}
    rows = []
    for dataset, summary in summaries.items():
        allowed = (('SSIM', 'PSNR', 'LPIPS', 'E00', 'UCIQE') if dataset == 'UIEB'
                   else ('UIQM', 'UCIQE', 'URanker', 'Entropy', 'NIQE'))
        for metric in allowed:
            sheet = ('Sheet2' if dataset != 'UIEB' else ('Sheet3' if metric == 'UCIQE' else 'Sheet1'))
            matches = [r for r in workbook['measurements'] if r['sheet'] == sheet
                       and r['group'] == names[dataset]
                       and r['metric'].strip().rstrip('↑↓').strip().casefold() == metric.casefold()]
            if len(matches) > 1:
                raise ValueError(f'Ambiguous workbook entry for {dataset}/{metric}')
            ref = matches[0] if matches else None
            if not ref and metric not in ('LPIPS', 'E00'):
                raise ValueError(f'Missing workbook reference for {dataset}/{metric}')
            value = summary[metric]
            delta = value - ref['value'] if ref else None
            better = None if ref is None else (delta < 0 if metric in ('LPIPS','E00','NIQE') else delta > 0)
            rows.append(dict(dataset=dataset, metric=metric, candidate=value,
                             mambacue=ref['value'] if ref else None, difference=delta,
                             numerical_win=better, workbook_cell=f"{sheet}!{ref['cell']}" if ref else None))
    return rows


def score(run, arm, audit_dir, workbook_reference, output):
    from .benchmark_mambacue import main as benchmark_candidate
    from .score_saved_lpips import evaluate as evaluate_lpips

    checkpoint, epoch = verify_checkpoint(run, arm, audit_dir)
    workbook = json.loads(workbook_reference.read_text(encoding='utf-8'))
    benchmark_candidate(output, student_checkpoint=checkpoint, uieb_audit=audit_dir / 'audit.json')
    manifest = Path('/home/user/MambaCue/supplement_20261112/protocol/UIEB_test.json')
    evaluate_lpips(output / 'UIEB', manifest)
    summaries = {name:json.loads((output / name / 'summary.json').read_text())
                 for name in ('UIEB', 'CH60', 'U45')}
    summaries['UIEB']['LPIPS'] = json.loads((output / 'UIEB/lpips.json').read_text())['LPIPS']
    report = dict(checkpoint_sha256=file_hash(checkpoint), arm=arm, epoch=epoch,
                  training_dataset='UIEB', evaluation_size=256,
                  workbook_sha256=workbook['sha256'], image_counts={k:v['count'] for k,v in summaries.items()},
                  rows=workbook_comparison(summaries, workbook),
                  interpretation='Numerical comparison to workbook entries; comparability requires matching test and metric protocols.')
    (output / 'workbook_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    score_ch60_sift(checkpoint, audit_dir, workbook, output)


def score_ch60_sift(checkpoint, audit_dir, workbook, output):
    """Match the CH60 input geometry, quantization and SIFT settings of Sheet6."""
    import cv2
    import numpy as np
    import torch

    from ..models.selective_student import build_student

    images = json.loads((audit_dir / 'audit.json').read_text())['no_reference_fingerprints']['CH60']
    if len(images) != 60:
        raise ValueError('Expected the frozen 60-image CH60 inventory')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model = build_student(state['config']['student'])
    model.load_state_dict(state['student'])
    model.cuda().eval()
    results = []
    folder = Path('/home/user/UnderWaterDataset/CH60/test')
    detector = cv2.SIFT_create()
    with torch.inference_mode():
        for name, fingerprints in sorted(images.items()):
            path = folder / name
            if file_hash(path) != fingerprints[0]:
                raise ValueError(f'CH60 input changed: {name}')
            image = cv2.imread(str(path))
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda().float() / 255
            prediction = model(x)[0].permute(1, 2, 0).cpu().numpy()
            enhanced = np.rint(np.clip(prediction, 0, 1) * 255).astype(np.uint8)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            points, _ = detector.detectAndCompute(gray, None)
            results.append(dict(id=name, keypoints=len(points)))
    reference, = (row for row in workbook['measurements'] if row['sheet'] == 'Sheet6' and row['cell'] == 'B11')
    score = sum(row['keypoints'] for row in results) / len(results)
    summary = dict(dataset='CH60', images=len(results), checkpoint_sha256=file_hash(checkpoint),
                   opencv=cv2.__version__, candidate_training_dataset='UIEB',
                   workbook_training_dataset='LSUI', workbook_cell='Sheet6!B11',
                   workbook_sha256=workbook['sha256'], workbook_keypoints=reference['value'],
                   candidate_keypoints=score, per_image=results)
    (output / 'ch60_sift.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('arm')
    for name in ('audit_dir', 'workbook_reference', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    score(args.run.resolve(), args.arm, args.audit_dir.resolve(), args.workbook_reference.resolve(),
          args.output.resolve())
