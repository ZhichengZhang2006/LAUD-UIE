import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from ..datasets import PairedDataset, read_manifest, validate_splits
from ..models import Student, KnowledgeViews, Controller
from ..models.teacher import build_teacher
from ..models.degradation import controller_inputs, image_statistics
from ..losses import ReconstructionLoss, metrics
from ..runtime import seed_all, append_json, file_hash
from .utility import estimate_utility


def strategy_weights(experiment, x, utility=None, controller=None, view=None):
    weights = x.new_zeros(x.shape[0], 3)
    if experiment == 'E0':
        return weights
    if experiment == 'E1':
        weights[:, KnowledgeViews.names.index(view)] = 1
        return weights
    if experiment == 'E2':
        return weights.fill_(1/3)
    if experiment == 'E3':
        stats = image_statistics(x)
        # Explicit heuristic baseline, not a calibrated physical degradation model.
        need = torch.stack([stats[:, :3].std(1, unbiased=False),
                            (stats[:, 6]-.5).abs(),
                            1-(stats[:, 8]+stats[:, 9]).clamp(0, 1)], 1)
        return need / need.sum(1, keepdim=True).clamp_min(1e-12)
    if experiment in ('E4', 'E5'):
        return controller.weights(utility)
    raise ValueError(experiment)


@torch.no_grad()
def validate(student, loader, device):
    student.eval()
    sums = {'psnr': 0., 'ssim': 0., 'mse': 0.}
    count = 0
    for batch in loader:
        x, target = batch['input'].to(device), batch['target'].to(device)
        for key, values in metrics(student(x), target).items():
            sums[key] += values.sum().item()
        count += x.shape[0]
    return {key: value/count for key, value in sums.items()}


def cycle(loader):
    while True:
        yield from loader


def train(config, output_dir, resume=None):
    seed_all(config['seed'])
    device = torch.device(config.get('device', 'cuda'))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=bool(resume))  # Resume only an explicitly selected run.
    (out/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    split_rows = {name: read_manifest(path, require_target=name != 'test') for name, path in config['data'].items() if path}
    validate_splits(split_rows)
    if not all(k in split_rows for k in ('train', 'val')):
        raise ValueError('train and val manifests are required')
    (out/'provenance.json').write_text(json.dumps({
        'torch': torch.__version__, 'device': str(device),
        'teacher_mode': config['teacher']['mode'],
        'distillation_with_checkpoint': config['experiment'] != 'E0' and config['teacher']['mode'] == 'checkpoint',
        'oracle_debug': config['experiment'] != 'E0' and config['teacher']['mode'] == 'oracle',
        'manifest_sha256': {k: file_hash(v) for k, v in config['data'].items() if v},
        'splits': split_rows,
        'teacher_sha256': file_hash(config['teacher']['checkpoint']) if config['teacher']['mode'] == 'checkpoint' else None,
    }, indent=2), encoding='utf-8')
    def loader(name, training=False, batch_size=None):
        # Independent generator means utility sampling never perturbs training shuffle.
        generator = torch.Generator().manual_seed(config['seed']+list(split_rows).index(name))
        dataset = PairedDataset(split_rows[name], config['crop_size'] if training else None, training)
        return DataLoader(dataset, batch_size=batch_size or (config['batch_size'] if training else 1),
                          shuffle=training, num_workers=0, generator=generator)
    train_loader, val_loader = loader('train', True), loader('val')
    student = Student(config['student']['width']).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config['lr'])
    experiment = config['experiment']
    teacher = build_teacher(config['teacher'], device) if experiment != 'E0' else None
    views, rec_loss = KnowledgeViews().to(device), ReconstructionLoss(config.get('ssim_weight', .2)).to(device)
    dynamic = experiment in ('E4', 'E5')
    uc = config.get('utility', {})
    controller = Controller(student.feature_dim, threshold=uc.get('threshold', 0.) if experiment == 'E5' else 0.).to(device) if dynamic else None
    ctrl_optimizer = torch.optim.Adam(controller.parameters(), lr=uc.get('lr', 1e-3)) if dynamic else None
    if dynamic:
        if uc.get('interval', 10) <= 0 or uc.get('virtual_interval', 100) <= 0:
            raise ValueError('Utility intervals must be positive')
        utility_train, utility_eval = cycle(loader('utility_train', batch_size=1)), cycle(loader('utility_eval', batch_size=1))
    start_epoch, step = 0, 0
    best = -float('inf')
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        student.load_state_dict(checkpoint['student'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if controller is not None and checkpoint.get('controller') is not None:
            controller.load_state_dict(checkpoint['controller'])
            ctrl_optimizer.load_state_dict(checkpoint['controller_optimizer'])
        start_epoch, step = checkpoint['epoch'] + 1, checkpoint['step']
        history = out/'epochs.jsonl'
        if history.exists():
            for line in history.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    best = max(best, json.loads(line)['val']['psnr'])
    def teacher_output(x, target):
        return target.detach() if config['teacher']['mode'] == 'oracle' else teacher(x)
    for epoch in range(start_epoch, config['epochs']):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        student.train()
        totals = {'loss': 0., 'rec': 0., 'kd': 0.}
        count = 0
        for batch in train_loader:
            # Only train-internal splits are consumed here; val/test never reach this path.
            utility_log = None
            if dynamic and step % uc.get('interval', 10) == 0:
                ub, eb = next(utility_train), next(utility_eval)
                ux, ut = ub['input'].to(device), ub['target'].to(device)
                ex, et = eb['input'].to(device), eb['target'].to(device)
                with torch.no_grad():
                    student.eval()
                    up, uf = student(ux, True)
                    ds, ss = controller_inputs(ux, up, uf)
                labels = estimate_utility(student, ux, teacher_output(ux, ut), ex, et, views, rec_loss,
                                          eta=uc.get('eta', .001), virtual=step % uc.get('virtual_interval', 100) == 0)
                # Signed cosine is the bounded, scale-stable label; dot/norm/gain remain recorded.
                prediction = controller(ds, ss)
                controller_loss = F.mse_loss(prediction, labels['cosine'][None])
                ctrl_optimizer.zero_grad(set_to_none=True)
                controller_loss.backward()
                ctrl_optimizer.step()
                utility_log = {'step': step, 'id': ub['id'][0], 'eval_id': eb['id'][0],
                               'prediction_before_update': prediction.detach().cpu().tolist()[0],
                               'controller_loss': controller_loss.item(),
                               **{k: v.cpu().tolist() for k, v in labels.items()}}
                append_json(out/'utility.jsonl', utility_log)
                student.train()
            x, target = batch['input'].to(device), batch['target'].to(device)
            prediction, features = student(x, True)
            rec = rec_loss(prediction, target)
            utility = None
            if dynamic:
                ds, ss = controller_inputs(x, prediction, features)
                with torch.no_grad():
                    utility = controller(ds, ss)
            weights = strategy_weights(experiment, x, utility, controller, config.get('view'))
            if dynamic and step < uc.get('warmup_steps', 100):
                weights = torch.zeros_like(weights)
            per_view = views(prediction, teacher_output(x, target)) if experiment != 'E0' else x.new_zeros(x.shape[0], 3)
            kd = (per_view*weights.detach()).sum(1).mean()
            loss = rec + config['kd_weight']*kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = x.shape[0]
            count += n
            for key, value in [('loss', loss), ('rec', rec), ('kd', kd)]:
                totals[key] += value.item()*n
            append_json(out/'steps.jsonl', {'step': step, 'epoch': epoch, 'ids': batch['id'],
                        'weights': weights.cpu().tolist(), 'view_losses': per_view.detach().cpu().tolist(),
                        'utility': utility.cpu().tolist() if utility is not None else None})
            step += 1
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter()-started
        scores = validate(student, val_loader, device)
        row = {'epoch': epoch, 'step': step, **{k: v/count for k, v in totals.items()},
               'val': scores, 'train_seconds': train_seconds,
               'peak_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
        append_json(out/'epochs.jsonl', row)
        print(json.dumps(row), flush=True)
        state = {'student': student.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'step': step, 'config': config, 'controller': controller.state_dict() if controller else None,
                 'controller_optimizer': ctrl_optimizer.state_dict() if ctrl_optimizer else None}
        torch.save(state, out/'last.pt')
        if scores['psnr'] > best:
            best = scores['psnr']
            torch.save(state, out/'best.pt')
    return student
