"""Report validation scores using the identical resize for every training arm."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest
from ..models.selective_student import build_student
from ..runtime import file_hash
from ..trainers.engine import validate


def score(run, manifest):
    checkpoint = run / 'best.pt'
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if file_hash(manifest) != file_hash(state['config']['data']['val']):
        raise ValueError('Only the training validation manifest is accepted')
    model = build_student(state['config']['student'])
    model.load_state_dict(state['student'], strict=True)
    torch.set_num_threads(2)
    rows = read_manifest(manifest)
    data = PairedDataset(rows, resize=256)
    loader = DataLoader(data, batch_size=8, shuffle=False)
    result = {
        'val_manifest_sha256': file_hash(manifest),
        'checkpoint_sha256': file_hash(checkpoint),
        'selected_epoch': state['epoch'] + 1,
        'sample_count': len(data),
        'geometry': 'PIL bilinear full image 256x256',
        'metrics': validate(model, loader, 'cpu'),
    }
    with (run / 'validation_256.json').open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('manifest', type=Path)
    args = p.parse_args()
    score(args.run, args.manifest)
