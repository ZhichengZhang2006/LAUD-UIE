"""Matched training screening with a fresh MambaCue arm and no test access."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest, validate_splits
from ..losses import ReconstructionLoss
from ..models.knowledge_views import lab_chroma
from ..models.selective_student import build_student
from ..runtime import append_json, file_hash, seed_all
from ..trainers.engine import validate


MODELS = ('packed_rgb', 'packed_bands', 'mambacue')
EXPOSURE_ARM = 'packed_exposure'
CONTEXT_ARM = 'packed_context'
PACKED_ARMS = ('packed_rgb', 'packed_bands', EXPOSURE_ARM, CONTEXT_ARM)


def loss_by_band(prediction, reference):
    """Compare cell-mean color and signed within-cell detail in the same loss."""
    error = F.pixel_unshuffle(prediction - reference, 2).unflatten(1, (3, 4))
    mean = error.mean(dim=2, keepdim=True)
    # The residual term supervises all three channels, including chromatic edges.
    detail = (error - mean).abs().mean()
    chroma = (lab_chroma(F.avg_pool2d(prediction, 2))
              - lab_chroma(F.avg_pool2d(reference, 2))).abs().mean()
    return .25 * detail + .15 * chroma


def loss_by_exposure(prediction, reference):
    """Give separate supervision to image exposure and spatial lighting error."""
    neutral = (prediction - reference).mean(dim=1, keepdim=True)
    image_bias = neutral.mean(dim=(2, 3), keepdim=True)
    lighting = F.avg_pool2d(neutral, 16) - image_bias
    return .25 * image_bias.abs().mean() + .25 * lighting.abs().mean()


def subset(rows, limit, seed):
    order = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())
    return order[:limit] if limit else order


def student_config(arm, width):
    if arm == 'mambacue':
        return {'architecture':'mambacue', 'width':32}
    return {'architecture':'packed', 'width':width, 'phase_residual':True,
            'global_context':arm == CONTEXT_ARM}


def load_network(arm, mamba_root, width):
    if arm == 'mambacue':
        sys.path.insert(0, str(mamba_root))
        from model import myModel
        return myModel(use_checkpoint=True)
    return build_student(student_config(arm, width))


def atomic_checkpoint(data, path):
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def train(arm, fit, validation, output, args):
    seed_all(args.seed)
    device = torch.device('cuda:0')
    output.mkdir()
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(PairedDataset(fit, augment=True, resize=256), batch_size=2,
                              shuffle=True, generator=generator, num_workers=2, pin_memory=True)
    val_loader = DataLoader(PairedDataset(validation, resize=256), batch_size=1, num_workers=2)
    network = load_network(arm, args.mamba, args.width).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=2e-5)
    reconstruction = ReconstructionLoss(ssim_weight=.2)
    best = -float('inf')
    step = 0
    for epoch in range(args.epochs):
        network.train()
        running_loss = 0.
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        for pair in train_loader:
            x, target = (pair[field].to(device, non_blocking=True) for field in ('input', 'target'))
            optimizer.zero_grad(set_to_none=True)
            prediction = network(x)
            loss = reconstruction(prediction, target)
            if arm in ('packed_bands', EXPOSURE_ARM, CONTEXT_ARM):
                loss = loss + loss_by_band(prediction, target)
            if arm in (EXPOSURE_ARM, CONTEXT_ARM):
                loss = loss + loss_by_exposure(prediction, target)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'{arm}: non-finite loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
            optimizer.step()
            running_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(json.dumps({'arm':arm, 'step':step, 'epoch':epoch+1}), flush=True)
        val = validate(network, val_loader, device)
        report = {'arm':arm, 'epoch':epoch+1, 'steps':step, 'val':val,
                  'training_loss':running_loss/len(train_loader),
                  'seconds':time.monotonic()-started, 'peak_allocated':torch.cuda.max_memory_allocated(device)}
        append_json(output/'epochs.jsonl', report)
        print(json.dumps(report), flush=True)
        scheduler.step()
        state = {'student':network.state_dict(), 'arm':arm, 'epoch':epoch+1, 'val':val,
                 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                 'config':{'student':student_config(arm, args.width),
                           'data':{'train':str(args.train),'val':str(args.val)}}}
        atomic_checkpoint(state, output/'last.pt')
        if val['psnr'] > best:
            best = val['psnr']
            atomic_checkpoint(state, output/'best.pt')


def run(args):
    # Set before the first CUDA call, so the deterministic Lab projection can use cuBLAS.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    fit = read_manifest(args.train)
    validation = read_manifest(args.val)
    validate_splits({'train':fit, 'val':validation})
    fit = subset(fit, args.train_limit, args.seed)
    validation = subset(validation, args.val_limit, args.seed)
    # Fail before creating a run directory if this runtime cannot acquire CUDA.
    torch.zeros(1, device='cuda:0')
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        'train_sha256':file_hash(args.train), 'val_sha256':file_hash(args.val),
        'train_ids':[row['id'] for row in fit], 'val_ids':[row['id'] for row in validation],
        'mamba_source':{name:file_hash(args.mamba/name) for name in ('model.py','encoder.py','prior.py','output_heads.py')},
        'epochs':args.epochs, 'seed':args.seed,
        'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(),
        'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'arms':list(args.arms), 'train_size':256, 'batch':2, 'student_width':args.width,
    }
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    if args.historical_mamba:
        opt = args.mamba/'supplement_20261112/protocol/historical_checkpoints.json'
        historical = json.loads(opt.read_text(encoding='utf-8'))['LSUI']
        weights = Path(historical['path'])
        if file_hash(weights) != historical['sha256']:
            raise ValueError('Historical MambaCue state has an unexpected hash')
        mamba = load_network('mambacue', args.mamba, args.width).cuda().eval()
        mamba.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        val_loader = DataLoader(PairedDataset(validation, resize=256), batch_size=1, num_workers=2)
        record = {'mambacue_sha256':historical['sha256'],
                  'val_ids':provenance['val_ids'], 'val':validate(mamba, val_loader, 'cuda:0')}
        (args.output/'historical_mambacue.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record['val']), flush=True)
        del mamba
        torch.cuda.empty_cache()
    for arm in args.arms:
        try:
            train(arm, fit, validation, args.output/arm, args)
        except RuntimeError as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower():
                (args.output/arm/'STOPPED_OOM.json').write_text(json.dumps({'arm':arm, 'error':str(exc), 'retry':False}))
                raise SystemExit(42) from exc
            raise
        torch.cuda.empty_cache()
    print('MATCHED_QUALITY_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('train','val','output'):
        parser.add_argument(arg, type=Path)
    parser.add_argument('--mamba', type=Path, default=Path('/home/user/MambaCue'))
    parser.add_argument('--train-limit', type=int, default=1536)
    parser.add_argument('--val-limit', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arms', nargs='+', choices=(*PACKED_ARMS, 'mambacue'), default=MODELS)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--historical-mamba', action='store_true')
    run(parser.parse_args())
