"""Capture fixed-shape CUDA inference and compare against ordinary inference."""
import argparse
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


def capture(model, input_shape, device):
    """Return the graph, its static input, and its reusable output tensor."""
    static = torch.rand(input_shape, device=device)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(10):
            model(static)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph):
        output = model(static)
    cost = {'capture_seconds':time.perf_counter() - started,
            'capture_extra_reserved_bytes':torch.cuda.memory_reserved() - reserved,
            'capture_extra_allocated_bytes':torch.cuda.memory_allocated() - allocated}
    return graph, static, output, cost


@torch.inference_mode()
def benchmark(root, output):
    if output.exists():
        raise FileExistsError(output)
    folder = root / 'standard/free'
    state = checked_checkpoints(folder)['last']
    checkpoint = torch.load(folder / 'last.pt', weights_only=True, map_location='cpu')
    protocol = json.loads((root / 'standard/protocol.json').read_text())
    validation = Path(protocol['parameters']['val'])
    if file_hash(validation) != protocol['val_sha256'] or len(protocol['val_ids']) != 128:
        raise ValueError('Expected the fixed 128-image validation inventory')
    rows = {row['id']:row for row in read_manifest(validation)}
    dataset = PairedDataset([rows[k] for k in protocol['val_ids']], resize=256)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    model = build_student(checkpoint['config']['student']).to(device).eval()
    model.load_state_dict(checkpoint['student'])
    model.set_inference_scan_batch(2)
    gpu_before = gpu_state()
    graph, static, replay_output, cost = capture(model, (1,3,256,256), device)

    def replay(x):
        static.copy_(x)
        graph.replay()
        return replay_output

    largest_delta = 0.
    for sample in DataLoader(dataset, batch_size=1, num_workers=2):
        x = sample['input'].to(device)
        reference = model(x)
        result = replay(x)
        torch.testing.assert_close(reference, result, rtol=1e-5, atol=2e-5)
        largest_delta = max(largest_delta, (reference-result).abs().max().item())

    image = dataset[0]['input'].unsqueeze(0).to(device)
    modes = {'eager':model, 'cuda_graph':replay}
    for mode in modes.values():
        for _ in range(10):
            mode(image)
    samples = {name:[] for name in modes}
    for index in range(60):
        order = list(modes) if index % 2 == 0 else list(reversed(modes))
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            modes[name](image)
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
    gains = [plain-captured for plain, captured in zip(samples['eager'], samples['cuda_graph'])]
    report = dict(checkpoint_sha256=state['sha256'], epoch=state['epoch'], model='standard/free',
                  batch=1, image_size=256, precision='FP32', scan_batch=2, tf32=False,
                  hardware=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  shared_gpu=True, gpu_before=gpu_before, gpu_after=gpu_state(),
                  parity_images=len(dataset), validation_manifest_sha256=protocol['val_sha256'],
                  largest_rgb_delta=largest_delta, gpu_graph=cost, samples_ms=samples,
                  median_time_ms={mode:median(times) for mode, times in samples.items()},
                  median_paired_gain_ms=median(gains))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(report['median_time_ms'], report['largest_rgb_delta'], cost, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    benchmark(args.root.resolve(), args.output.resolve())
