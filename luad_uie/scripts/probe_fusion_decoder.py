"""Measure decoder low-rank approximations on a frozen 1.0 development checkpoint."""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest
from ..models.selective_student import build_student
from ..runtime import file_hash
from ..trainers.engine import validate


@torch.no_grad()
def factorize(kernel, rank):
    if kernel.groups != 1 or kernel.bias is not None:
        raise ValueError('Expected a dense, bias-free decoder convolution')
    matrix = kernel.weight.flatten(1)
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    basis = nn.Conv2d(kernel.in_channels, rank, kernel.kernel_size,
                      stride=kernel.stride, padding=kernel.padding, bias=False)
    mixing = nn.Conv2d(rank, kernel.out_channels, 1, bias=False)
    basis.weight.copy_(right[:rank].reshape_as(basis.weight))
    mixing.weight.copy_((left[:, :rank] * values[:rank])[:, :, None, None])
    return nn.Sequential(basis, mixing)


def reduce_decoder(model, fraction):
    decoder = model.backbone.decoder
    for block in (decoder.dec3, decoder.dec2, decoder.dec1):
        for sequence, index in ((block.up, 1), (block.block.block, 0), (block.block.block, 3)):
            layer = sequence[index]
            rank = max(8, 8 * round(layer.out_channels * fraction / 8))
            sequence[index] = factorize(layer, rank)


@torch.inference_mode()
def interleaved_timing(models, image):
    for model in models.values():
        for _ in range(10):
            model(image)
    torch.cuda.synchronize()
    timings = {key: [] for key in models}
    keys = list(models)
    for trial in range(36):
        start = trial % len(keys)
        for key in keys[start:] + keys[:start]:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            models[key](image)
            torch.cuda.synchronize()
            timings[key].append((time.perf_counter() - begin) * 1000)
    return timings


def run(checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((checkpoint.parent.parent / 'protocol.json').read_text())
    val = Path(protocol['parameters']['val'])
    if file_hash(val) != protocol['val_sha256']:
        raise ValueError('Validation manifest changed')
    records = {r['id']: r for r in read_manifest(val)}
    dataset = PairedDataset([records[k] for k in protocol['val_ids']], resize=256)
    loader = DataLoader(dataset, batch_size=1, num_workers=2)
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    base = build_student(state['config']['student']).eval()
    base.load_state_dict(state['student'])
    models = {'dense': base}
    for fraction in (.75, .5):
        net = copy.deepcopy(base)
        reduce_decoder(net, fraction)
        models[str(fraction)] = net
    models = {k:v.cuda().eval() for k, v in models.items()}
    timings = interleaved_timing(models, dataset[0]['input'].unsqueeze(0).cuda())
    results = {}
    for name, model in models.items():
        scores = validate(model, loader, 'cuda')
        results[name] = dict(val=scores, parameters=sum(p.numel() for p in model.parameters()),
                             latency_ms=statistics.median(timings[name]), timing_samples_ms=timings[name])
    if abs(results['dense']['val']['psnr'] - state['val']['psnr']) > .001:
        raise ValueError('Dense reference failed to reproduce its development score')
    output.write_text(json.dumps(dict(checkpoint=file_hash(checkpoint), validation_ids=protocol['val_ids'],
                                      gpu=torch.cuda.get_device_name(), shared_gpu=True,
                                      fp32=True, batch_size=1, image_size=256, results=results), indent=2))
    for k, r in results.items():
        print(k, r['val'], r['parameters'], r['latency_ms'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args.checkpoint.resolve(), args.output.resolve())
