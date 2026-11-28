"""Freeze the existing 2711/128 LSUI development split and official project test list."""
import argparse
import csv
import json
from pathlib import Path

from data_v22 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path('/home/user/luad_uie_prototype_20261031/stages/cf_clean_20261119/data/CF_LSUI')
    protocol = Path('/home/user/luad_fusion_1_0/formal100/full_lsui_free/protocol.json')
    official = Path('/home/user/MambaCue/supplement_20261112/protocol/LSUI_test.json')
    frozen = json.loads(protocol.read_text())
    manifests = {}
    for split, expected in [('train', 2711), ('val', 128)]:
        path = source / f'{split}.csv'
        if sha256(path) != frozen[f'{split}_sha256']:
            raise ValueError(f'LSUI {split} source changed since the recorded development run')
        with path.open(newline='', encoding='utf-8-sig') as stream:
            rows = {row['id']: row for row in csv.DictReader(stream)}
        ids = frozen[f'{split}_ids']
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f'Unexpected LSUI {split} membership')
        manifests[split] = [rows[name] for name in ids]
    manifests['test'] = [
        {'id': Path(row['name']).stem, 'input': row['input'], 'target': row['target']}
        for row in json.loads(official.read_text())
    ]
    if len(manifests['test']) != 428:
        raise ValueError('Unexpected LSUI test membership')
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in manifests.items():
        path = args.output / f'{split}.csv'
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, ['id', 'input', 'target'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    print({split: len(rows) for split, rows in manifests.items()}, flush=True)


if __name__ == '__main__':
    main()
