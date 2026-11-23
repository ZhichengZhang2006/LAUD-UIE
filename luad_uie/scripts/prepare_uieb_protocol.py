"""Prepare a UIEB training split with no exact overlap with the evaluation images."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from ..runtime import file_hash


def image_fingerprints(path):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert('RGB')
        pixels = hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode())
        pixels.update(rgb.tobytes())
    return hashlib.sha256(data).hexdigest(), pixels.hexdigest()


def prepare(protocol, datasets, output):
    no_reference = {}
    evaluation_hashes = set()
    for name in ('CH60', 'U45'):
        images = sorted(p for p in (datasets/name/'test').iterdir()
                        if p.suffix.lower() in ('.jpg', '.png', '.bmp', '.jpeg'))
        if len(images) != {'CH60': 60, 'U45': 45}[name]:
            raise ValueError(f'Unexpected {name} count: {len(images)}')
        values = {image.name: image_fingerprints(image) for image in images}
        no_reference[name] = values
        evaluation_hashes.update(value for pair in values.values() for value in pair)

    seen = {}
    rows = {}
    excluded = {}
    manifests = {}
    for split in ('test', 'val', 'train'):
        path = protocol/f'UIEB_{split}.json'
        manifests[split] = file_hash(path)
        rows[split], excluded[split] = [], []
        for source in json.loads(path.read_text()):
            input_path, target_path = Path(source['input']), Path(source['target'])
            inp, ref = image_fingerprints(input_path), image_fingerprints(target_path)
            if inp[0] != source['input_sha256'] or ref[0] != source['target_sha256']:
                raise ValueError(f"UIEB content changed: {source['name']}")
            fingerprints = set(inp + ref)
            if split != 'test' and fingerprints.intersection(evaluation_hashes):
                excluded[split].append({'id': source['name'], 'reason': 'no_reference_overlap'})
                continue
            other = sorted({seen[fp] for fp in fingerprints if fp in seen and seen[fp] != split})
            if other:
                excluded[split].append({'id': source['name'], 'reason': 'overlap_with_' + '_'.join(other)})
                continue
            rows[split].append({'id': source['name'], 'input': input_path, 'target': target_path})
            seen.update({fp: split for fp in fingerprints})
    if any(not part for part in rows.values()):
        raise ValueError('The UIEB overlap audit emptied a split')
    output.mkdir(parents=True, exist_ok=False)
    for split, values in rows.items():
        with (output/f'{split}.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=('id', 'input', 'target'))
            writer.writeheader()
            writer.writerows(values)
    audit = {'source_manifests': manifests, 'sizes': {k: len(v) for k, v in rows.items()},
             'excluded': excluded, 'no_reference_fingerprints': no_reference,
             'csv_sha256': {split: file_hash(output/f'{split}.csv') for split in rows}}
    (output/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'sizes': audit['sizes'], 'excluded': excluded}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('protocol', type=Path)
    p.add_argument('datasets', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    prepare(args.protocol, args.datasets, args.output)
