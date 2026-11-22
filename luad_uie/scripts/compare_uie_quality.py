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
    if dataset == 'LSUI':
        record = json.loads((folder / 'lpips.json').read_text())
        values = {row['id']: row['lpips_vgg'] for row in record['records']}
        if set(values) != set(scores) or len(record['records']) != len(scores):
            raise ValueError(f'{folder}: incomplete LPIPS')
        for image_id, value in values.items():
            scores[image_id]['LPIPS'] = value
    return scores


def compare(baseline, candidate):
    raw = {}
    provenance = [json.loads((p/'provenance.json').read_text()) for p in (baseline, candidate)]
    for key in ('manifest_sha256', 'preprocessing', 'source_sha256'):
        if provenance[0][key] != provenance[1][key]:
            raise ValueError(f'Benchmark implementations or inputs differ: {key}')
    for dataset in ('LSUI', 'U45', 'CH60'):
        first, second = (results(p, dataset) for p in (baseline, candidate))
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
    return {'quality_wins':wins, 'quality_endpoints':len(rows), 'meets_80_percent_quality': wins>=8,
            'results':rows, 'checkpoint_sha256':[r['checkpoint']['sha256'] for r in provenance]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    report = compare(args.baseline, args.candidate)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
