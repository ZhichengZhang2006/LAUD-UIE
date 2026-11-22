"""Measure exposure and structure on a fixed validation split."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest
from ..losses import metrics
from ..models.selective_student import build_student
from ..runtime import file_hash


def components(prediction, reference, cell=16):
    error = (prediction - reference).double()
    neutral = error.mean(1, keepdim=True)
    opponent = error - neutral
    terms = {}
    for kind, color in [('neutral', neutral), ('chroma', opponent)]:
        overall = color.mean((2, 3), keepdim=True)
        blocks = torch.nn.functional.avg_pool2d(color, cell, cell)
        expanded = blocks.repeat_interleave(cell, 2).repeat_interleave(cell, 3)
        for scale, value in [('global', overall), ('local', blocks-overall), ('detail', color-expanded)]:
            terms[f'{scale}_{kind}'] = value.square().mean((1, 2, 3))
    full = error.square().mean((1, 2, 3))
    torch.testing.assert_close(full, sum(terms.values()), atol=1e-12, rtol=1e-9)
    return terms


def load_student(checkpoint, mamba_root, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('arm') == 'mambacue':
        sys.path.insert(0, str(mamba_root))
        from model import myModel
        model = myModel()
    else:
        model = build_student(state['config']['student'])
    model.load_state_dict(state['student'])
    return model.to(device).eval()


@torch.inference_mode()
def score(args):
    manifest = read_manifest(args.val)
    provenance = json.loads(args.protocol.read_text(encoding='utf-8'))
    if provenance['val_sha256'] != file_hash(args.val):
        raise ValueError('Validation manifest differs from training provenance')
    by_id = {row['id']:row for row in manifest}
    ids = provenance['val_ids']
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicated validation IDs')
    loader = DataLoader(PairedDataset([by_id[i] for i in ids], resize=256), batch_size=4)
    rows = []
    network = load_student(args.checkpoint, args.mamba, args.device)
    for inputs in loader:
        source, target = (inputs[key].to(args.device) for key in ('input', 'target'))
        predicted = network(source)
        scores = {**metrics(predicted, target), **components(predicted, target)}
        rows.extend({'id':identifier, **{key:float(value[i]) for key,value in scores.items()}}
                    for i,identifier in enumerate(inputs['id']))
    result = {'val_sha256':file_hash(args.val), 'checkpoint_sha256':file_hash(args.checkpoint),
              'validation_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
              'count':len(rows),
              'summary':{k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k!='id'},
              'per_image':rows}
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    for name in ('checkpoint','val','protocol','output'):
        cli.add_argument(name, type=Path)
    cli.add_argument('--device', default='cpu')
    cli.add_argument('--mamba', type=Path, default=Path('/home/user/MambaCue'))
    args = cli.parse_args()
    torch.set_num_threads(2)
    score(args)
