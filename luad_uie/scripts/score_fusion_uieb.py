"""Score one completed UIEB fusion arm against the frozen workbook entries."""
import argparse
import csv
import json
from pathlib import Path

from ..runtime import file_hash


def verify_checkpoint(run, arm, audit_dir):
    import torch

    if arm not in (run / 'COMPLETE').read_text().split():
        raise ValueError('Requested arm is incomplete')
    protocol = json.loads((run / 'protocol.json').read_text())
    budget = protocol['parameters']['epochs']
    history = [json.loads(line)['epoch'] for line in (run / arm / 'epochs.jsonl').read_text().splitlines()]
    if (budget < 100 or protocol['parameters'].get('smoke_test')
            or history != list(range(1, budget + 1))):
        raise ValueError('Formal scoring requires the full declared budget of at least 100 epochs')
    audit = json.loads((audit_dir / 'initialization.json').read_text())
    checkpoint = run / arm / 'best.pt'
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if (state['arm'] != arm or state['config']['student']['architecture'] != 'backbone_fusion'
            or state['backbone_sha256'] != audit['backbone_sha256']
            or protocol['weights_sha256'] != state['backbone_sha256']):
        raise ValueError('Unexpected model or pretraining source')
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
                  workbook_sha256=workbook['sha256'], image_counts={k:v['count'] for k,v in summaries.items()},
                  rows=workbook_comparison(summaries, workbook),
                  interpretation='Numerical comparison to workbook entries; comparability requires matching test and metric protocols.')
    (output / 'workbook_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('arm')
    for name in ('audit_dir', 'workbook_reference', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    score(args.run.resolve(), args.arm, args.audit_dir.resolve(), args.workbook_reference.resolve(),
          args.output.resolve())
