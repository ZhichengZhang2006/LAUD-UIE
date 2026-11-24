"""Measure a lower-precision deployment path for one fixed 100-epoch model."""
import argparse
import json
import time
from pathlib import Path
from statistics import mean, median

import torch
from torch.utils.data import DataLoader

from ..datasets import PairedDataset, read_manifest
from ..losses import metrics
from ..models.selective_student import build_student
from ..runtime import file_hash
from .profile_fusion_workbook import gpu_state
from .summarize_fusion_100 import checked_checkpoints


@torch.inference_mode()
def benchmark(root, output):
    if output.exists():
        raise FileExistsError(output)
    folder = root / 'standard/free'
    final = checked_checkpoints(folder)['last']
    saved = torch.load(folder / 'last.pt', weights_only=True, map_location='cpu')
    protocol = json.loads((root / 'standard/protocol.json').read_text())
    val = Path(protocol['parameters']['val'])
    if file_hash(val) != protocol['val_sha256'] or len(protocol['val_ids']) != 128:
        raise ValueError('Expected the fixed 128-image validation inventory')
    rows = {row['id']:row for row in read_manifest(val)}
    dataset = PairedDataset([rows[name] for name in protocol['val_ids']], resize=256)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This comparison requires hardware BF16')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_student(saved['config']['student']).cuda().eval()
    model.load_state_dict(saved['student'])
    model.set_inference_scan_batch(2)

    def mixed(x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(x)

    modes = {'fp32':model, 'bf16':mixed, 'bf16_backbone':model.forward_bf16_backbone}
    gpu_before = gpu_state()
    per_image = []
    for batch in DataLoader(dataset, batch_size=1, num_workers=2):
        x, reference = batch['input'].cuda(), batch['target'].cuda()
        restored = {name:forward(x).float() for name, forward in modes.items()}
        report = {'id':batch['id'][0]}
        for name, prediction in restored.items():
            if not torch.isfinite(prediction).all():
                raise ValueError(f'Non-finite {name} output on image {batch["id"][0]}')
            report[name] = {k:v.item() for k,v in metrics(prediction, reference).items()}
            report[name]['max_rgb_delta'] = (prediction-restored['fp32']).abs().max().item()
        per_image.append(report)

    sample = dataset[0]['input'].unsqueeze(0).cuda()
    samples = {name:[] for name in modes}
    peaks = dict.fromkeys(modes, 0)
    for fn in modes.values():
        for _ in range(10):
            fn(sample)
    names = list(modes)
    for trial in range(60):
        offset = trial % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            modes[name](sample)
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
            peaks[name] = max(peaks[name], torch.cuda.max_memory_allocated() - resident)

    quality = {}
    for name in ('bf16', 'bf16_backbone'):
        quality[name] = {}
        for metric in ('psnr', 'ssim'):
            change = [row[name][metric] - row['fp32'][metric] for row in per_image]
            quality[name][metric] = dict(mean_delta=mean(change), median_delta=median(change),
                                         worst_delta=min(change), worst_abs_delta=max(map(abs, change)))
    summary = dict(epoch=final['epoch'], checkpoint_sha256=final['sha256'], shape=[1,3,256,256],
                   scan_batch=2, tf32=False, parity_images=len(per_image),
                   val_manifest_sha256=protocol['val_sha256'], hardware=torch.cuda.get_device_name(),
                   torch_version=str(torch.__version__), shared_gpu=True,
                   gpu_before=gpu_before, gpu_after=gpu_state(), samples_ms=samples,
                   median_ms={name:median(values) for name, values in samples.items()},
                   peak_extra_bytes=peaks, quality=quality, per_image=per_image)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(summary['median_ms'], summary['peak_extra_bytes'], quality, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    benchmark(args.root.resolve(), args.output.resolve())
