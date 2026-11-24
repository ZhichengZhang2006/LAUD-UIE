"""Compare inference scan batching on one audited 100-epoch checkpoint."""
import argparse
import itertools
import json
import time
from pathlib import Path
from statistics import median

import torch
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest
from ..models.selective_student import build_student
from ..runtime import file_hash
from .profile_fusion_workbook import gpu_state
from .summarize_fusion_100 import checked_checkpoints


SIZES = (1, 2, 4)


@torch.inference_mode()
def compare(root, output):
    if output.exists():
        raise FileExistsError(output)
    directory = root / 'standard/free'
    final = checked_checkpoints(directory)['last']
    checkpoint = directory / 'last.pt'
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    protocol = json.loads((root / 'standard/protocol.json').read_text())
    val = Path(protocol['parameters']['val'])
    if file_hash(val) != protocol['val_sha256'] or len(protocol['val_ids']) != 128:
        raise ValueError('Expected the 128-image validation inventory')
    inventory = {row['id']:row for row in read_manifest(val)}
    pairs = PairedDataset([inventory[name] for name in protocol['val_ids']], resize=256)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_student(state['config']['student']).cuda().eval()
    model.load_state_dict(state['student'])
    before = gpu_state()

    maxima = dict.fromkeys(SIZES, 0.)
    for batch in DataLoader(pairs, batch_size=1, num_workers=2):
        x = batch['input'].cuda()
        model.set_inference_scan_batch(1)
        reference = model(x)
        for size in SIZES[1:]:
            model.set_inference_scan_batch(size)
            candidate = model(x)
            torch.testing.assert_close(candidate, reference, atol=2e-5, rtol=1e-5)
            maxima[size] = max(maxima[size], (candidate-reference).abs().max().item())

    # Rotate every permutation to balance ordering during shared-GPU activity.
    test_image = pairs[0]['input'].unsqueeze(0).cuda()
    for size in SIZES:
        model.set_inference_scan_batch(size)
        for _ in range(10):
            model(test_image)
    torch.cuda.synchronize()
    orders = list(itertools.permutations(SIZES)) * 10
    timings = {size:[] for size in SIZES}
    peaks = dict.fromkeys(SIZES, 0)
    for order in orders:
        for size in order:
            model.set_inference_scan_batch(size)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model(test_image)
            torch.cuda.synchronize()
            timings[size].append((time.perf_counter()-started)*1000)
            peaks[size] = max(peaks[size], torch.cuda.max_memory_allocated()-allocated)
    after = gpu_state()
    result = {}
    for size in SIZES:
        timings_ms = timings[size]
        paired_deltas = [candidate-control for candidate, control in zip(timings_ms, timings[1])]
        result[size] = dict(median_ms=median(timings_ms), p90_ms=sorted(timings_ms)[53],
                            paired_median_delta_ms=median(paired_deltas), measurements_ms=timings_ms,
                            peak_extra_bytes=peaks[size], max_rgb_delta=maxima[size])
    report = dict(epoch=final['epoch'], checkpoint_sha256=final['sha256'], image_size=256, batch=1,
                  precision='FP32', tf32=False, parity_validation_images=len(pairs),
                  val_manifest_sha256=protocol['val_sha256'], shared_gpu=True,
                  gpu_before=before, gpu_after=after, interleaved_orders=orders, results=result)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print({size:(score['median_ms'], score['peak_extra_bytes'], score['max_rgb_delta'])
           for size, score in result.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('formal_root', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    compare(args.formal_root.resolve(), args.output.resolve())
