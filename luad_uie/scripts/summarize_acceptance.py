"""Tie quality, downstream, and runtime results to one checkpoint and protocol."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def build_report(quality, downstream, timing):
    baseline, candidate = quality['checkpoint_sha256']
    report = {
        'checkpoint_sha256': {'mambacue': baseline, 'candidate': candidate},
        'quality': {'wins': quality['quality_wins'], 'endpoints': quality['quality_endpoints'],
                    'pass': quality['meets_80_percent_quality']},
        'boundary': {'pass': None},
        'detection': {'pass': None},
        'speed': {'pass': None},
    }
    boundary = read(downstream / 'suim.json')
    if boundary:
        if (boundary['mambacue'], boundary['candidate']) != (baseline, candidate):
            raise ValueError('SUIM checkpoint differs from the quality comparison')
        a, b = (boundary['metrics'][name]['f1'] for name in ('mambacue', 'candidate'))
        report['boundary'] = {'mambacue_f1': a, 'candidate_f1': b, 'pass': b > a}
    manifest = []
    detection = []
    for name, expected in (('mambacue', baseline), ('candidate', candidate)):
        arm = downstream / name
        record, provenance = read(arm/'metrics.json'), read(arm/'provenance.json')
        if record and provenance:
            if provenance['checkpoint'] != expected:
                raise ValueError(f'RUOD {name} checkpoint mismatch')
            if record['annotation_sha256'] != provenance['annotations']:
                raise ValueError(f'RUOD {name} annotations mismatch')
            detection.append(record)
            manifest.append((provenance['detector'], provenance['annotations'], record['images']))
    if len(detection) == 2:
        if manifest[0] != manifest[1]:
            raise ValueError('RUOD detector or sample set differs')
        a, b = (row['coco']['AP'] for row in detection)
        report['detection'] = {'mambacue_ap': a, 'candidate_ap': b, 'pass': b > a,
                               'images': manifest[0][2]}
    if timing and timing.get('checkpoint_sha256'):
        if timing['checkpoint_sha256'] != [baseline, candidate]:
            raise ValueError('Timing checkpoint differs from the quality comparison')
        comparisons = {}
        for size in ('256', '512'):
            row = timing['measurements'][size]
            a, b = (row[name]['median_ms'] for name in ('mambacue', 'candidate'))
            comparisons[size] = {'mambacue_ms': a, 'candidate_ms': b, 'faster': b < a}
        report['speed'] = {'sizes': comparisons, 'shared_gpu': timing['shared_gpu'],
                           'pass': all(row['faster'] for row in comparisons.values())}
    gates = [report[name]['pass'] for name in ('quality', 'boundary', 'detection', 'speed')]
    report['all_measurement_gates_pass'] = all(gate is True for gate in gates)
    report['novelty_and_ablation_evidence_required'] = True
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('quality', 'downstream', 'timing', 'output'):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    report = build_report(read(a.quality), a.downstream, read(a.timing))
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False), flush=True)
