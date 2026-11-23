"""Audit paired test sets against both stages of fusion training."""
import argparse
import json
from pathlib import Path

from ..datasets import read_manifest
from ..runtime import file_hash
from .benchmark_paired_transfer import DATASETS, DATA_ROOT, pairs
from .prepare_uieb_protocol import image_fingerprints


def prepare(train, val, protocol, output):
    development = {'lsui_train': train, 'lsui_val': val,
                   **{f'uieb_{split}':protocol / f'UIEB_{split}.json' for split in ('train', 'val')}}
    sources = set()
    for manifest in development.values():
        rows = read_manifest(manifest) if manifest.suffix == '.csv' else json.loads(manifest.read_text())
        sources.update(Path(row[key]).resolve() for row in rows for key in ('input', 'target'))
    used = {fp for path in sources for fp in image_fingerprints(path)}
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for dataset in ('LSUI', *DATASETS):
        if dataset == 'LSUI':
            manifest = protocol / 'LSUI_test.json'
            rows = json.loads(manifest.read_text())
        else:
            manifest, rows = pairs(dataset, protocol, DATA_ROOT)
        kept, excluded = [], []
        for row in rows:
            inputs = image_fingerprints(Path(row['input']))
            targets = image_fingerprints(Path(row['target']))
            if (row.get('input_sha256', inputs[0]) != inputs[0]
                    or row.get('target_sha256', targets[0]) != targets[0]):
                raise ValueError(f'Test image differs from manifest: {row["name"]}')
            if used.intersection(inputs + targets):
                excluded.append(row['name'])
                continue
            kept.append(dict(name=row['name'], input=row['input'], target=row['target'],
                             input_sha256=inputs[0], target_sha256=targets[0]))
        if not kept:
            raise ValueError(f'{dataset}: all test pairs overlap development data')
        out = output / f'{dataset}.json'
        out.write_text(json.dumps(kept, indent=2), encoding='utf-8')
        inventory[dataset] = dict(source_sha256=file_hash(manifest), manifest_sha256=file_hash(out),
                                  count=len(kept), excluded=excluded)
        print(f'{dataset}: kept {len(kept)}, excluded {len(excluded)}', flush=True)
    result = dict(development_sha256={key:file_hash(path) for key, path in development.items()},
                  datasets=inventory)
    (output / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('train', 'val', 'protocol', 'output'):
        parser.add_argument(field, type=Path)
    prepare(**vars(parser.parse_args()))
