"""Test frozen LSUI-trained enhancers on external paired datasets."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import torch
from skimage.color import deltaE_ciede2000, rgb2lab

from ..datasets import read_manifest
from .benchmark_mambacue import digest, load_rgb
from .benchmark_uie_downstream import MAMBA, models, write
from .prepare_uieb_protocol import image_fingerprints

DATASETS = ('UIEB', 'UFO', 'EUVP-s', 'EUVP-d')
DATA_ROOT = Path('/home/user/UnderWaterDataset')


def pairs(dataset, protocol, root):
    if dataset in ('UIEB', 'UFO'):
        manifest = protocol/f'{dataset}_test.json'
        rows = json.loads(manifest.read_text())
    else:
        folder = root/dataset/'test'
        manifest = folder/'data_list.txt'
        names = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
        if any(Path(name).name != name for name in names):
            raise ValueError(f'Unexpected paired filename in {manifest}')
        rows = [{'name': name, 'input': str(folder/'input'/name), 'target': str(folder/'GT'/name)}
                for name in names]
    if not rows or len({row['name'] for row in rows}) != len(rows):
        raise ValueError(f'Empty or ambiguous manifest: {manifest}')
    return manifest, rows


def development_fingerprints(checkpoint, protocol):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    paths = set()
    manifest_sha = {}
    for split in ('train', 'val'):
        manifest = Path(state['config']['data'][split])
        manifest_sha[f'candidate_{split}'] = digest(manifest)
        reference = protocol/f'LSUI_{split}.json'
        manifest_sha[f'reference_{split}'] = digest(reference)
        for rows in (read_manifest(manifest), json.loads(reference.read_text())):
            paths.update(Path(row[key]).resolve() for row in rows for key in ('input', 'target'))
    hashes = set()
    for path in sorted(paths):
        hashes.update(image_fingerprints(path))
    return hashes, manifest_sha


@torch.inference_mode()
def run(checkpoint, output, selected):
    torch.set_num_threads(2)
    from lpips import LPIPS
    protocol = MAMBA/'supplement_20261112/protocol'
    known, acknowledged_manifests = development_fingerprints(checkpoint, protocol)
    networks, baseline_sha = models(checkpoint)
    sys.path.insert(0, str(MAMBA))
    from utils import metrics
    perceptual = LPIPS(net='vgg', version='0.1').cuda().eval()
    output.mkdir(parents=True, exist_ok=False)
    provenance = {'training_dataset': 'LSUI', 'working_size': 256,
                  'mambacue_run_training_inventory_known': False,
                  'checkpoints': {'mambacue': baseline_sha, 'candidate': digest(checkpoint)},
                  'excluded_against_manifest_sha256': acknowledged_manifests}
    write(output/'provenance.json', provenance)
    for dataset in selected:
        manifest, rows = pairs(dataset, protocol, DATA_ROOT)
        result, excluded = [], []
        for row in rows:
            inp, ref = (Path(row[key]) for key in ('input', 'target'))
            inp_fp, ref_fp = image_fingerprints(inp), image_fingerprints(ref)
            for key, values in [('input', inp_fp), ('target', ref_fp)]:
                if f'{key}_sha256' in row and row[f'{key}_sha256'] != values[0]:
                    raise ValueError(f'Unexpected {dataset} file content: {row["name"]}')
            if known.intersection(inp_fp + ref_fp):
                excluded.append(row['name'])
                continue
            x, y = load_rgb(inp), load_rgb(ref)
            target = y.permute(0, 2, 3, 1).cpu().numpy()
            target_lab = rgb2lab(target[0])
            images = {}
            for arm, net in networks.items():
                image = net(x).clamp(0, 1)
                pred = image.permute(0, 2, 3, 1).cpu().numpy()
                images[arm] = {'SSIM': float(metrics.getSSIM(pred, target, 1.)),
                               'PSNR': float(metrics.getPSNR(pred, target, 1.)),
                               'LPIPS': float(perceptual(image, y, normalize=True).item()),
                               'E00': float(deltaE_ciede2000(rgb2lab(pred[0]), target_lab).mean()),
                               'UCIQE': float(metrics.getUCIQE(pred))}
            result.append({'id': row['name'], 'input_sha256': inp_fp[0], 'target_sha256': ref_fp[0],
                           'metrics': images})
        if not result:
            raise ValueError(f'No remaining {dataset} pairs after overlap audit')
        summary = {arm: {name: mean(row['metrics'][arm][name] for row in result)
                         for name in result[0]['metrics'][arm]} for arm in networks}
        info = {'dataset': dataset, 'manifest_sha256': digest(manifest), 'initial_count': len(rows),
                'count': len(result), 'excluded': excluded, 'summary': summary, 'images': result}
        write(output/f'{dataset}.json', info)
        print(dataset, len(result), len(excluded), json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = p.parse_args()
    run(args.checkpoint.resolve(), args.output.resolve(), args.datasets)
