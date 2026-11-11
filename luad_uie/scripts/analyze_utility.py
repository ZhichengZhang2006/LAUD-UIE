"""Offline controller audit on NEW train-internal candidates and references."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..datasets import read_manifest, validate_splits, PairedDataset
from ..models import Student, Controller, KnowledgeViews
from ..models.teacher import build_teacher
from ..models.degradation import controller_inputs
from ..losses import ReconstructionLoss
from ..trainers.utility import estimate_utility
from ..runtime import append_json, seed_all


def correlation(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rank(a):
    # Average ranks for ties; no scipy dependency.
    a = np.asarray(a)
    return np.array([(np.sum(a < v)+(np.sum(a == v)-1)/2) for v in a])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--candidates', required=True, help='Unused TRAIN-INTERNAL audit manifest')
    p.add_argument('--references', required=True, help='Disjoint TRAIN-INTERNAL audit reference manifest')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--eta', type=float, default=.001)
    args = p.parse_args(argv)
    run = Path(args.run)
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    config = state['config']
    seed_all(config['seed'])
    if state['controller'] is None:
        raise ValueError('This audit requires an E4/E5 controller checkpoint')
    splits = json.loads((run/'provenance.json').read_text())['splits']
    candidates, references = read_manifest(args.candidates), read_manifest(args.references)
    validate_splits({**splits, 'audit_candidates': candidates, 'audit_references': references})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    student = Student(config['student']['width']).to(args.device).eval()
    student.load_state_dict(state['student'])
    controller = Controller(student.feature_dim).to(args.device).eval()
    controller.load_state_dict(state['controller'])
    teacher = build_teacher(config['teacher'], args.device)
    rec, views = ReconstructionLoss(config.get('ssim_weight', .2)), KnowledgeViews()
    refs, rows = PairedDataset(references), []
    for i, sample in enumerate(PairedDataset(candidates)):
        reference = refs[i % len(refs)]
        x, target = sample['input'][None].to(args.device), sample['target'][None].to(args.device)
        ex, et = reference['input'][None].to(args.device), reference['target'][None].to(args.device)
        with torch.no_grad():
            output, features = student(x, True)
            prediction = controller(*controller_inputs(x, output, features))[0]
            teacher_rgb = target if config['teacher']['mode'] == 'oracle' else teacher(x)
        labels = estimate_utility(student, x, teacher_rgb, ex, et, views, rec, args.eta, True)
        row = {'id': sample['id'], 'reference_id': reference['id'], 'prediction': prediction.cpu().tolist(),
               **{k: v.cpu().tolist() for k, v in labels.items()}}
        rows.append(row)
        append_json(out/'audit.jsonl', row)
    prediction = np.array([r['prediction'] for r in rows])
    cosine = np.array([r['cosine'] for r in rows])
    dot = np.array([r['dot'] for r in rows])
    gain = np.array([r['virtual_gain'] for r in rows])
    summary = {'teacher_mode': config['teacher']['mode'], 'samples': len(rows), 'per_view': {}}
    for k, name in enumerate(views.names):
        summary['per_view'][name] = {
            'prediction_cosine_mse': float(np.mean((prediction[:, k]-cosine[:, k])**2)),
            'negative_gain_accuracy': float(np.mean((prediction[:, k] < 0) == (gain[:, k] < 0))),
            'negative_gain_prevalence': float(np.mean(gain[:, k] < 0)),
            'prediction_gain_spearman': correlation(rank(prediction[:, k]), rank(gain[:, k])),
            'dot_gain_pearson': correlation(dot[:, k], gain[:, k]),
            'cosine_gain_pearson': correlation(cosine[:, k], gain[:, k]),
        }
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
