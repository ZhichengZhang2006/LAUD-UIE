"""Interleave both models within one process to compare GPU inference latency."""
import argparse
import json
from pathlib import Path
from statistics import median
import torch
from benchmark_uie_downstream import models


@torch.inference_mode()
def run(checkpoint, output):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    networks, _ = models(checkpoint)
    report = {}
    for size in (256, 512):
        x = torch.rand(1,3,size,size, device='cuda', generator=torch.Generator(device='cuda').manual_seed(42))
        names = ('mambacue', 'candidate')
        samples = {name: [] for name in names}
        for _ in range(8):
            for name in names:
                networks[name](x)
        for i in range(40):
            for name in names if i % 2 == 0 else names[::-1]:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                networks[name](x)
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
        report[size] = {k:{'median_ms':median(v), 'p90_ms':sorted(v)[35],
                           'parameters':sum(p.numel() for p in networks[k].parameters())}
                        for k,v in samples.items()}
    with output.open('x') as stream:
        json.dump({'model_shape': 'batch=1, RGB, float32, TF32 off', 'shared_gpu':True,
                   'device':torch.cuda.get_device_name(),'measurements':report},stream,indent=2)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('candidate', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    run(args.candidate, args.output)
