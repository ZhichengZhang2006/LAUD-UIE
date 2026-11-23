"""Compare identical image IDs and report the fixed ten quality endpoints."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def results(directory, dataset):
    folder = directory / dataset
    summary = json.loads((folder / 'summary.json').read_text())
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    ids = [row['id'] for row in rows]
    if not ids or len(ids) != summary['count'] or len(set(ids)) != len(ids):
        raise ValueError(f'{folder}: incomplete or duplicate image scores')
    scores = {row['id']: row for row in rows}
    if dataset in ('LSUI', 'UIEB'):
        record = json.loads((folder / 'lpips.json').read_text())
        values = {row['id']: row['lpips_vgg'] for row in record['records']}
        if set(values) != set(scores) or len(record['records']) != len(scores):
            raise ValueError(f'{folder}: incomplete LPIPS')
        for image_id, value in values.items():
            scores[image_id]['LPIPS'] = value
    return scores


def matching_provenance(directories):
    records = [json.loads((path/'provenance.json').read_text()) for path in directories]
    for key in ('manifest_sha256', 'preprocessing', 'source_sha256'):
        if records[0][key] != records[1][key]:
            raise ValueError(f'Benchmark implementations or inputs differ: {key}')
    return records


def compare(baseline, candidate, unpaired=None):
    raw = {}
    paired = (baseline, candidate)
    provenance = matching_provenance(paired)
    if unpaired is not None:
        uieb = matching_provenance(unpaired)
        audit = uieb[0].get('uieb_audit_sha256')
        if not audit or any(row.get('training_dataset') != 'UIEB' or row.get('uieb_audit_sha256') != audit
                            for row in uieb):
            raise ValueError('U45/CH60 require UIEB-trained weights and the same audited data')
    else:
        uieb = None  # Archived combined scoring is available for historical reports.
    for dataset in ('LSUI', 'U45', 'CH60'):
        directories = paired if dataset == 'LSUI' else (unpaired or paired)
        first, second = (results(p, dataset) for p in directories)
        if set(first) != set(second):
            raise ValueError(f'Dataset IDs differ: {dataset}')
        raw[dataset] = first, second
    metrics = [
        ('LSUI', 'SSIM', 1), ('LSUI', 'PSNR', 1), ('LSUI', 'LPIPS', -1),
        ('LSUI', 'E00', -1), ('LSUI', 'UCIQE', 1), ('unpaired', 'UIQM', 1),
        ('unpaired', 'UCIQE', 1), ('unpaired', 'URanker', 1),
        ('unpaired', 'Entropy', 1), ('unpaired', 'NIQE', -1),
    ]
    rows = []
    for subset, metric, direction in metrics:
        datasets = ('LSUI',) if subset == 'LSUI' else ('U45', 'CH60')
        baseline_score, candidate_score = [
            mean(mean(row[metric] for row in raw[dataset][side].values()) for dataset in datasets)
            for side in range(2)
        ]
        if not (math.isfinite(baseline_score) and math.isfinite(candidate_score)):
            raise ValueError(f'Invalid measurement: {subset}/{metric}')
        rows.append({'subset':subset, 'metric':metric, 'mambacue':baseline_score,
                     'candidate':candidate_score, 'wins':direction*(candidate_score-baseline_score)>0})
    wins = sum(row['wins'] for row in rows)
    report = {'quality_wins':wins, 'quality_endpoints':len(rows),
              'meets_80_percent_quality':wins>=8 and uieb is not None,
              'results':rows, 'checkpoint_sha256':[r['checkpoint']['sha256'] for r in provenance]}
    if uieb is not None:
        report['unpaired_training_dataset'] = 'UIEB'
        report['unpaired_checkpoint_sha256'] = [record['checkpoint']['sha256'] for record in uieb]
        report['uieb_audit_sha256'] = audit
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--unpaired-baseline', type=Path)
    p.add_argument('--unpaired-candidate', type=Path)
    args = p.parse_args()
    if bool(args.unpaired_baseline) != bool(args.unpaired_candidate):
        p.error('Both UIEB benchmark directories are required')
    unpaired = (args.unpaired_baseline, args.unpaired_candidate) if args.unpaired_baseline else None
    report = compare(args.baseline, args.candidate, unpaired)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
