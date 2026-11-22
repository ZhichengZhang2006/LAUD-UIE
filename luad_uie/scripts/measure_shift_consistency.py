"""Check whether one-pixel shifts change predictions on frozen validation images."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest
from ..losses.phase_consistency import interior, shift
from ..models.knowledge_views import luminance
from ..runtime import file_hash
from .score_validation_decomposition import load_student


def assessment(base, shifted, reference):
    difference = (base - shifted).abs().mean(dim=(1,2,3))
    original, translated, target = map(luminance, (base, shifted, reference))
    edge_sum = torch.zeros_like(difference)
    edge_count = torch.zeros_like(difference)
    for axis in (2,3):
        edges = torch.diff(target, dim=axis).abs() >= .02
        change = torch.diff(original - translated, dim=axis).abs()
        edge_sum += (change * edges).sum(dim=(1,2,3))
        edge_count += edges.sum(dim=(1,2,3))
    return difference, edge_sum / edge_count.clamp_min(1), edge_count


@torch.inference_mode()
def measure(args):
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if protocol['val_sha256'] != file_hash(args.manifest):
        raise ValueError('Unexpected validation split')
    by_id = {row['id']:row for row in read_manifest(args.manifest)}
    ids = protocol['val_ids'][:args.limit]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Expected unique validation IDs')
    loader = DataLoader(PairedDataset([by_id[k] for k in ids], resize=256), batch_size=1)
    compact = load_student(args.compact, args.mamba_root, args.device)
    original = load_student(args.gated, args.mamba_root, args.device)
    known = json.loads((args.mamba_root/'supplement_20261112/protocol/historical_checkpoints.json')
                       .read_text(encoding='utf-8'))['LSUI']
    baseline = Path(known['path'])
    if file_hash(baseline) != known['sha256']:
        raise ValueError('Historical MambaCue checkpoint mismatch')
    sys.path.insert(0, str(args.mamba_root))
    from model import myModel
    mambacue = myModel().to(args.device).eval()
    mambacue.load_state_dict(torch.load(baseline, map_location='cpu', weights_only=True))
    networks = {'packed':compact, 'gated':original, 'mambacue':mambacue}
    torch.backends.cuda.matmul.allow_tf32 = False
    offsets = {'odd_x':(1,0), 'odd_y':(0,1), 'two_x':(2,0), 'stride_x':(8,0)}
    rows = []
    for pair in loader:
        image, target = (pair[key].to(args.device) for key in ('input','target'))
        views = {name:shift(image, *delta) for name,delta in offsets.items()}
        for name, network in networks.items():
            restored = interior(network(image))
            for offset, shifted in views.items():
                aligned = interior(network(shifted), *offsets[offset])
                rgb, edges, support = assessment(restored, aligned, interior(target))
                rows.append({'id':pair['id'][0],'arm':name,'offset':offset,
                             'rgb_mae':float(rgb[0]),'edge_disagreement':float(edges[0]),
                             'edge_count':int(support[0])})
    summary = {}
    for name in networks:
        summary[name] = {}
        for offset in offsets:
            selected = [r for r in rows if r['arm']==name and r['offset']==offset]
            summary[name][offset] = {
                'rgb_mae':float(np.mean([r['rgb_mae'] for r in selected])),
                'edge_disagreement':float(np.mean([r['edge_disagreement'] for r in selected if r['edge_count']])),
            }
    report = {
        'validation_manifest_sha256':file_hash(args.manifest),
        'ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(), 'images':len(ids),
        'checkpoint_sha256':{'packed':file_hash(args.compact), 'gated':file_hash(args.gated), 'mambacue':known['sha256']},
        'offsets_xy':offsets, 'comparison_crop_border':32, 'summary':summary, 'per_image':rows,
    }
    with args.output.open('x',encoding='utf-8') as out:
        json.dump(report,out,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    for arg in ('manifest','protocol','compact','gated','output'):
        cli.add_argument(arg,type=Path)
    cli.add_argument('--mamba-root', type=Path, default=Path('/home/user/MambaCue'))
    cli.add_argument('--limit', type=int, default=32)
    cli.add_argument('--device', default='cuda:0')
    opts = cli.parse_args()
    torch.set_num_threads(2)
    measure(opts)
