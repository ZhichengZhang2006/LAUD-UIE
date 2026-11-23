"""Score formally trained fusion candidates on audited paired benchmarks."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import torch
from skimage.color import deltaE_ciede2000, rgb2lab

from ..runtime import file_hash
from .benchmark_mambacue import load_rgb
from .summarize_fusion_100 import collect


METRICS = ('SSIM', 'PSNR', 'LPIPS', 'E00', 'UCIQE')
PROTOCOL = Path('/home/user/MambaCue/supplement_20261112/protocol')


def reference_rows(dataset, scores, workbook, coverage):
    group = 'UFO-120' if dataset == 'UFO' else dataset
    rows = []
    for name in METRICS:
        sheet = 'Sheet3' if name == 'UCIQE' and dataset in ('LSUI', 'UIEB', 'UFO') else 'Sheet1'
        matches = [item for item in workbook['measurements'] if item['sheet'] == sheet
                   and item['group'] == group and item['metric'].rstrip('↑↓').strip() == name]
        if len(matches) > 1:
            raise ValueError(f'Multiple workbook references: {dataset}/{name}')
        ref = matches[0] if matches else None
        rows.append(dict(metric=name, candidate=scores[name], mambacue=ref['value'] if ref else None,
                         workbook_cell=f"{sheet}!{ref['cell']}" if ref else None,
                         full_source_coverage=coverage))
    return rows


@torch.inference_mode()
def score(run, arm, audit_dir, workbook_file, output):
    from lpips import LPIPS
    from ..models.selective_student import build_student

    completed = collect(run, full_data=arm == 'full_lsui_free/free')
    selected = completed['arms'][arm]['best']
    checkpoint = run / arm / 'best.pt'
    if file_hash(checkpoint) != selected['sha256']:
        raise ValueError('Candidate checkpoint changed')
    audited = json.loads((audit_dir / 'audit.json').read_text())
    for split in ('train', 'val'):
        if (audited['development_sha256'][f'lsui_{split}'] != completed[f'{split}_manifest_sha256']
                or audited['development_sha256'][f'uieb_{split}'] != file_hash(PROTOCOL / f'UIEB_{split}.json')):
            raise ValueError(f'Overlap audit and training protocol differ for {split}')
    workbook = json.loads(workbook_file.read_text(encoding='utf-8'))
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    model = build_student(state['config']['student']).cuda().eval()
    model.load_state_dict(state['student'])
    perceptual = LPIPS(net='vgg', version='0.1').cuda().eval()
    sys.path.insert(0, str(PROTOCOL.parents[1]))
    from utils import metrics

    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for dataset, meta in audited['datasets'].items():
        manifest = audit_dir / f'{dataset}.json'
        if file_hash(manifest) != meta['manifest_sha256']:
            raise ValueError(f'Changed audited manifest: {dataset}')
        per_image = []
        for row in json.loads(manifest.read_text()):
            for key in ('input', 'target'):
                if file_hash(row[key]) != row[f'{key}_sha256']:
                    raise ValueError(f'Changed image in {dataset}: {row["name"]}')
            x, y = load_rgb(row['input']), load_rgb(row['target'])
            image = model(x).clamp(0, 1)
            pred, target = (t.permute(0, 2, 3, 1).cpu().numpy() for t in (image, y))
            result = dict(id=row['name'], SSIM=float(metrics.getSSIM(pred, target, 1.)),
                          PSNR=float(metrics.getPSNR(pred, target, 1.)),
                          LPIPS=float(perceptual(image, y, normalize=True).item()),
                          E00=float(deltaE_ciede2000(rgb2lab(pred[0]), rgb2lab(target[0])).mean()),
                          UCIQE=float(metrics.getUCIQE(pred)))
            per_image.append(result)
        summary = {name:mean(row[name] for row in per_image) for name in METRICS}
        summaries[dataset] = dict(count=len(per_image), excluded=meta['excluded'],
                                  metrics=reference_rows(dataset, summary, workbook, not meta['excluded']))
        (output / f'{dataset}.json').write_text(json.dumps(per_image, allow_nan=False))
        print(dataset, len(per_image), summary, flush=True)
    summary = dict(training_dataset='LSUI', pretraining_dataset='UIEB', evaluation_size=256,
                   arm=arm, selected_epoch=selected['epoch'], checkpoint_sha256=selected['sha256'],
                   workbook_sha256=workbook['sha256'], audit_sha256=file_hash(audit_dir/'audit.json'),
                   datasets=summaries)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('arm')
    for field in ('audit_dir', 'workbook_file', 'output'):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    score(args.run.resolve(), args.arm, args.audit_dir.resolve(), args.workbook_file.resolve(),
          args.output.resolve())
