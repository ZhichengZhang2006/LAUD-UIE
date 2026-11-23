"""Freeze the historical 320/80 DUO rows and normalized YOLO labels."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ..runtime import file_hash


CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')
ALIASES = {'sea cucumber': 'holothurian', 'sea_cucumber': 'holothurian',
           'sea urchin': 'echinus', 'sea_urchin': 'echinus'}


def prepare(root, selection, output):
    selected = json.loads(selection.read_text())
    records = selected['records']
    if selected['seed'] != 42 or Counter(row['split'] for row in records) != {'train':320, 'test':80}:
        raise ValueError('Expected the historical DUO 320/80 selection at seed 42')
    if len({row['sha256'] for row in records}) != len(records):
        raise ValueError('Duplicate content in DUO selection')

    result = []
    annotation_hashes = {}
    for split in ('train', 'test'):
        path = root / 'annotations' / f'instances_{split}.json'
        annotation_hashes[split] = file_hash(path)
        annotations = json.loads(path.read_text())
        images = {row['id']: row for row in annotations['images']}
        labels = defaultdict(list)
        for item in annotations['annotations']:
            labels[item['image_id']].append(item)
        class_map = {}
        for item in annotations['categories']:
            name = item['name'].strip().lower()
            class_map[item['id']] = CLASSES.index(ALIASES.get(name, name))
        for upstream in records:
            if upstream['split'] != split:
                continue
            image_id = upstream['id']
            metadata = images[image_id]
            filename = Path(metadata['file_name'])
            image_root = root / 'images'
            options = (image_root / filename, image_root / split / filename,
                       image_root / f'{split}2017' / filename)
            source = next((p for p in options if p.is_file()), None)
            if source is None:
                raise FileNotFoundError(filename)
            if file_hash(source) != upstream['sha256']:
                raise ValueError(f'Image content differs from historical DUO: {source}')
            with Image.open(source) as image:
                width, height = image.size
            if (width, height) != (metadata['width'], metadata['height']):
                raise ValueError(f'Annotation geometry differs: {source}')
            boxes = []
            for item in labels[image_id]:
                x, y, w, h = map(float, item['bbox'])
                left, top = max(0, x), max(0, y)
                right, bottom = min(width, x + w), min(height, y + h)
                if item.get('iscrowd') or item.get('ignore') or right <= left or bottom <= top:
                    raise ValueError(f'Unsupported DUO annotation: {item["id"]}')
                boxes.append([class_map[item['category_id']], (left + right) / (2 * width),
                              (top + bottom) / (2 * height), (right - left) / width,
                              (bottom - top) / height])
            result.append(dict(**upstream, input=str(source.resolve()), yolo_labels=boxes))
    if any(len({box[0] for row in result if row['split'] == split for box in row['yolo_labels']}) != 4
           for split in ('train', 'test')):
        raise ValueError('Selected train and test splits must each contain all DUO classes')
    if len({(row['split'], row['id']) for row in result}) != len(records):
        raise ValueError('Expected distinct DUO sample IDs per split')
    report = dict(classes=CLASSES, selection_sha256=file_hash(selection),
                  annotation_sha256=annotation_hashes, records=result)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'selection', 'output'):
        parser.add_argument(argument, type=Path)
    prepare(**vars(parser.parse_args()))
