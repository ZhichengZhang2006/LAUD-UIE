"""Train only derived enhancement models. No full MambaCue evaluation path."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest, validate_splits
from ..losses import ReconstructionLoss
from ..losses.component_distillation import component_distillation
from ..losses.ordinal_structure import ordinal_structure, structure_metrics
from ..models.backbone_fusion import BackboneFusion
from ..models.scene_initialization import fit_components, initialize_head
from ..runtime import append_json, file_hash, seed_all
from ..trainers.engine import validate
from .train_matched_quality import atomic_checkpoint, subset


ARMS = ('free', 'centered', 'ordinal', 'fixed', 'utility')
BACKBONE_SHA256 = 'dc43cb10dd8fa4940b4f12eb0a315fe8d1c6c05f733bdd6da137c5be234239fc'


def train_arm(arm, args, fit, validation, provenance):
    seed_all(args.seed)
    out = args.output / arm
    out.mkdir()
    model = BackboneFusion(centered=arm != 'free')
    initial = torch.load(args.weights, weights_only=True, map_location='cpu')
    model.backbone.load_state_dict(initial, strict=True)
    model.cuda()
    teacher = None
    if arm in ('fixed', 'utility'):
        teacher_path = args.output / 'free' / 'best.pt'
        teacher = BackboneFusion(centered=False)
        teacher.load_state_dict(torch.load(teacher_path, weights_only=True, map_location='cpu')['student'])
        teacher.requires_grad_(False).eval().cuda()
        (out / 'teacher.sha256').write_text(file_hash(teacher_path))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(PairedDataset(fit, resize=256, augment=True), batch_size=args.batch,
                        shuffle=True, generator=generator, num_workers=2)
    val_loader = DataLoader(PairedDataset(validation, resize=256), batch_size=1, num_workers=2)
    if args.head_fit_images:
        model.eval()
        calibration = DataLoader(PairedDataset(fit[:args.head_fit_images], resize=256), batch_size=1)
        with torch.no_grad():
            def feature_pairs():
                for batch in calibration:
                    image = batch['input'].cuda()
                    yield model.backbone(image), batch['target'].cuda() - image
            initialize_head(model.head, fit_components(feature_pairs()))
            initial_scores = validate(model, val_loader, 'cuda', structure_metrics)
        (out / 'initial.json').write_text(json.dumps(initial_scores, indent=2))
    optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 2e-5},
                                   {'params': model.head.parameters(), 'lr': 2e-4}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=2e-6)
    reconstruction = ReconstructionLoss(ssim_weight=.2)
    best = -float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        training = torch.zeros((), device='cuda')
        routed = torch.zeros(2, device='cuda')
        seen = 0
        for batch in loader:
            x, y = batch['input'].cuda(), batch['target'].cuda()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = reconstruction(prediction, y)
            if arm in ('ordinal', 'fixed', 'utility'):
                loss = loss + .2 * ordinal_structure(prediction, x, y)
            if teacher is not None:
                with torch.no_grad():
                    taught = teacher(x)
                kd, weights = component_distillation(prediction, taught, y, gated=arm == 'utility')
                loss = loss + .1 * kd
                routed += weights.sum(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            seen += len(x)
            training += loss.detach() * len(x)
        scores = validate(model, val_loader, 'cuda', structure_metrics)
        row = dict(arm=arm, epoch=epoch, train_loss=(training / seen).item(), val=scores,
                   kd_weights=(routed / seen).tolist() if teacher else None)
        append_json(out / 'epochs.jsonl', row)
        print(json.dumps(row), flush=True)
        scheduler.step()
        state = dict(student=model.state_dict(), epoch=epoch, arm=arm, val=scores,
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     version='1.0', seed=args.seed, backbone_sha256=provenance['weights_sha256'],
                     config={'student': {'architecture': 'backbone_fusion', 'centered': arm != 'free'},
                             'experiment': arm, 'evaluation_resize': 256, 'seed': args.seed,
                             'epochs': args.epochs, 'batch_size': args.batch,
                             'teacher': {'mode': 'checkpoint' if teacher is not None else 'none'},
                             'data': {'train': str(args.train.resolve()), 'val': str(args.val.resolve())}})
        atomic_checkpoint(state, out / 'last.pt')
        if scores['psnr'] > best:
            best = scores['psnr']
            atomic_checkpoint(state, out / 'best.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'val', 'weights', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Allow a shorter engineering check, ineligible for performance comparison')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--val-limit', type=int, default=0)
    parser.add_argument('--head-fit-images', type=int, default=0,
                        help='Initialize the existing RGB head using this many training pairs')
    args = parser.parse_args()
    if args.epochs < 100 and not args.smoke_test:
        parser.error('Formal comparisons require at least 100 epochs per arm')
    if min(args.epochs, args.batch) < 1 or min(args.train_limit, args.val_limit) < 0:
        parser.error('Invalid budget or subset size')
    if len(set(args.arms)) != len(args.arms):
        parser.error('Duplicate experiment arm')
    if any(arm in ('fixed', 'utility') for arm in args.arms) and args.arms[0] != 'free':
        parser.error('Include free first to train the frozen teacher on the same split')
    fingerprint = file_hash(args.weights)
    if fingerprint != BACKBONE_SHA256:
        raise ValueError('Expected the UIEB EMA weights from the supplied backbone archive')
    fit, val = read_manifest(args.train), read_manifest(args.val)
    validate_splits({'train': fit, 'val': val})
    fit = subset(fit, args.train_limit, args.seed)
    val = subset(val, args.val_limit, args.seed)
    if args.head_fit_images and not 2 <= args.head_fit_images <= len(fit):
        parser.error('Head calibration requires between two images and the full training split')
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(version='1.0', weights_sha256=fingerprint, train_sha256=file_hash(args.train),
                      val_sha256=file_hash(args.val), train_ids=[x['id'] for x in fit],
                      val_ids=[x['id'] for x in val], head_fit_ids=[x['id'] for x in fit[:args.head_fit_images]],
                      gpu=torch.cuda.get_device_name(),
                      parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output / 'protocol.json').write_text(json.dumps(provenance, indent=2))
    for arm in args.arms:
        try:
            train_arm(arm, args, fit, val, provenance)
        except torch.cuda.OutOfMemoryError:
            (args.output / 'STOPPED_OOM').write_text(arm)
            raise
    (args.output / 'COMPLETE').write_text(' '.join(args.arms))


if __name__ == '__main__':
    main()
