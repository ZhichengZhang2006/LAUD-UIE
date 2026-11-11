import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def read_manifest(path, require_target=True):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    result = []
    for row in rows:
        if not row.get('id') or not row.get('input') or (require_target and not row.get('target')):
            raise ValueError('Manifest requires id,input,target columns (target optional for inference)')
        item = {'id': row['id']}
        for key in ('input', 'target'):
            item[key] = str((path.parent / row[key]).resolve()) if row.get(key) else None
            if item[key] and not Path(item[key]).is_file():
                raise FileNotFoundError(item[key])
        result.append(item)
    if len({r['id'] for r in result}) != len(result):
        raise ValueError('Duplicate image IDs within manifest')
    return result


def validate_splits(splits):
    """Reject ID or canonical file reuse across train/utility/val/test manifests."""
    seen_ids, seen_paths = {}, {}
    for split, rows in splits.items():
        for row in rows:
            if row['id'] in seen_ids and seen_ids[row['id']] != split:
                raise ValueError(f'Split leakage: ID {row["id"]}')
            seen_ids[row['id']] = split
            for key in ('input', 'target'):
                path = row.get(key)
                if not path:
                    continue
                if path in seen_paths and seen_paths[path] != split:
                    raise ValueError(f'Split leakage: path {path}')
                seen_paths[path] = split


class PairedDataset(Dataset):
    def __init__(self, rows, crop_size=None, augment=False):
        self.rows, self.crop_size, self.augment = rows, crop_size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        keys = ['input'] + (['target'] if row.get('target') else [])
        images = [Image.open(row[k]).convert('RGB') for k in keys]
        if len({im.size for im in images}) != 1:
            raise ValueError(f'Unaligned image sizes: {row["id"]}')
        if self.crop_size:
            size = self.crop_size
            w, h = images[0].size
            # Replicated border avoids geometric distortion for small pairs.
            if min(w, h) < size:
                images = [Image.fromarray(np.pad(np.asarray(im), ((0, max(0, size-h)), (0, max(0, size-w)), (0, 0)), mode='edge')) for im in images]
            w, h = images[0].size
            left, top = random.randint(0, w-size), random.randint(0, h-size)
            images = [im.crop((left, top, left+size, top+size)) for im in images]
        if self.augment:
            if random.random() < .5:
                images = [ImageOps.mirror(im) for im in images]
            if random.random() < .5:
                images = [ImageOps.flip(im) for im in images]
        tensors = {k: torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1).copy())/255 for k, im in zip(keys, images)}
        return {'id': row['id'], **tensors}
