"""Assemble ten fixed quality endpoints from candidate results and the frozen workbook."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


PAIRED = ('SSIM', 'PSNR', 'LPIPS', 'E00', 'UCIQE')
UNPAIRED = ('UIQM', 'UCIQE', 'URanker', 'Entropy', 'NIQE')
LOWER_IS_BETTER = {'LPIPS', 'E00', 'NIQE'}


def endpoints(paired, uieb, workbook, evidence):
    if paired['workbook_sha256'] != workbook['sha256'] or uieb['workbook_sha256'] != workbook['sha256']:
        raise ValueError('Candidate reports reference different workbooks')
    if paired['training_dataset'] != 'LSUI' or paired['evaluation_size'] != 256:
        raise ValueError('Expected the frozen 256×256 LSUI candidate results')
    if uieb['training_dataset'] != 'UIEB' or uieb['evaluation_size'] != 256:
        raise ValueError('Expected the frozen 256×256 UIEB candidate results')
    if any(uieb['image_counts'].get(name) != size for name, size in (('CH60', 60), ('U45', 45))):
        raise ValueError('Expected complete CH60 and U45 scores')
    original = {f"{r['sheet']}!{r['cell']}": r['value'] for r in workbook['measurements']}
    cells_by_metric = {(r['sheet'], r['group'], r['metric'].rstrip('↑↓').strip().casefold()):
                       f"{r['sheet']}!{r['cell']}" for r in workbook['measurements']}
    lsui = {row['metric']:row for row in paired['datasets']['LSUI']['metrics']}
    nr = {(row['dataset'],row['metric']):row for row in uieb['rows']}
    result = {}
    for track, names in (('LSUI', PAIRED), ('CH60_U45', UNPAIRED)):
        for name in names:
            key = f'{track}/{name}'
            rows = [lsui[name]] if track == 'LSUI' else [nr[dataset, name] for dataset in ('CH60', 'U45')]
            cells = [row['workbook_cell'] for row in rows]
            sheet = ('Sheet3' if name == 'UCIQE' else 'Sheet1') if track == 'LSUI' else 'Sheet2'
            groups = ('LSUI',) if track == 'LSUI' else ('C 60', 'U45')
            if cells != [cells_by_metric.get((sheet, group, name.casefold())) for group in groups]:
                raise ValueError(f'{key}: expected workbook metric and dataset cells')
            if any(row['mambacue'] != original.get(cell) for row, cell in zip(rows, cells)):
                raise ValueError(f'{key}: workbook reference and recorded value differ')
            candidate = mean(row['candidate'] for row in rows)
            if not math.isfinite(candidate):
                raise ValueError(f'{key}: non-finite score')
            baseline = mean(original[cell] for cell in cells) if all(cells) else None
            win = None if baseline is None else (candidate < baseline if name in LOWER_IS_BETTER else candidate > baseline)
            coverage = all(row.get('full_source_coverage', True) for row in rows)
            verified = win if coverage and evidence.get(key) else None
            result[key] = dict(candidate=candidate, mambacue=baseline, workbook_cells=cells,
                               numerical_win=win, verified_win=verified, full_source_coverage=coverage,
                               protocol_evidence=evidence.get(key))
    if set(evidence) - set(result) or any(not isinstance(note, str) or not note.strip()
                                         for note in evidence.values()):
        raise ValueError('Protocol evidence requires known endpoint keys and a source note')
    return result


def summarize(paired, uieb, workbook, evidence):
    rows = endpoints(paired, uieb, workbook, evidence)
    wins = sum(row['verified_win'] is True for row in rows.values())
    unknown = sum(row['verified_win'] is None for row in rows.values())
    gate = True if wins >= 8 else False if wins + unknown < 8 else None
    return dict(checkpoints={'LSUI': paired['checkpoint_sha256'], 'UIEB': uieb['checkpoint_sha256']},
                workbook_sha256=workbook['sha256'], quality_endpoints=rows,
                verified_wins=wins, unverified=unknown, required_wins=8, denominator=10,
                reaches_80_percent=gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('paired', 'uieb', 'workbook', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--protocol-evidence', type=Path,
                        help='Optional JSON object mapping verified endpoint names to protocol evidence')
    args = parser.parse_args()
    files = (args.paired, args.uieb, args.workbook)
    evidence = json.loads(args.protocol_evidence.read_text(encoding='utf-8')) if args.protocol_evidence else {}
    report = summarize(*(json.loads(path.read_text(encoding='utf-8')) for path in files), evidence)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
