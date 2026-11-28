"""Materialize the existing five-domain protocol for the post-ablation runs."""
import argparse
import csv
import json
import shutil
from pathlib import Path

from data_v22 import read_manifest, sha256


DATASETS = Path('/home/user/UnderWaterDataset')
SUPPLEMENT = Path('/home/user/MambaCue/supplement_20261112/protocol')
UIEB = Path('/home/user/luad_fusion_1_0/uieb_pretrain_audit')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def write_split(root, domain, split, rows):
    path = root / domain / f'{split}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['id', 'input', 'target'] if 'target' in rows[0] else ['id', 'input']
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def paired_directory(path):
    def files(folder):
        images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        by_id = {p.stem: p for p in images}
        if len(by_id) != len(images):
            raise ValueError(f'Ambiguous image stems in {folder}')
        return by_id

    inputs, targets = files(path / 'input'), files(path / 'GT')
    if not inputs or inputs.keys() != targets.keys():
        raise ValueError(f'Unmatched input/reference files in {path}')
    return [{'id': key, 'input': inputs[key], 'target': targets[key]} for key in sorted(inputs)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inventory = {}

    def register(domain, split, path, source, expected=None, paired=True):
        rows = read_manifest(path, paired, domain)
        if expected is not None and len(rows) != expected:
            raise ValueError(f'{domain}/{split}: expected {expected}, found {len(rows)}')
        inventory[f'{domain}/{split}'] = {
            'n': len(rows), 'csv_sha256': sha256(path), 'source': str(source)
        }

    for split, count in zip(('train', 'val', 'test'), (712, 88, 89)):
        source = UIEB / f'{split}.csv'
        target = root / 'UIEB' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        register('UIEB', split, target, source, count)

    for domain, label, counts in [('LSUI', 'LSUI', (3404, 411, 428)),
                                  ('UFO-120', 'UFO', (1197, 297, 120))]:
        for split, count in zip(('train', 'val', 'test'), counts):
            source = SUPPLEMENT / f'{label}_{split}.json'
            rows = [{key: row[source_key] for key, source_key in
                     [('id', 'name'), ('input', 'input'), ('target', 'target')]}
                    for row in json.loads(source.read_text(encoding='utf-8'))]
            path = write_split(root, domain, split, rows)
            register(domain, split, path, source, count)

    for domain in ('EUVP-s', 'EUVP-d'):
        for split in ('train', 'val', 'test'):
            source = DATASETS / domain / split
            path = write_split(root, domain, split, paired_directory(source))
            register(domain, split, path, source)

    for domain, label, count in [('C60', 'CH60', 60), ('U45', 'U45', 45), ('EUVP', 'EUVP', 330)]:
        source = DATASETS / label / 'test'
        rows = [{'id': p.name, 'input': p} for p in sorted(source.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        path = write_split(root, domain, 'test', rows)
        register(domain, 'test', path, source, count, paired=False)

    (root / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    print({key: row['n'] for key, row in inventory.items()})


if __name__ == '__main__':
    main()
