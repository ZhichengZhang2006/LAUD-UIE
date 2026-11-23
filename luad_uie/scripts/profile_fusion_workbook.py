"""Profile a completed UIEB candidate under the archived Sheet4 settings."""
import argparse
import copy
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from ..runtime import file_hash
from .score_fusion_uieb import verify_checkpoint


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.free,utilization.gpu',
                             '--format=csv'], capture_output=True, text=True, check=True)
    return result.stdout.strip()


@torch.inference_mode()
def measure(model, image):
    from thop import profile
    import thop

    profiled = copy.deepcopy(model)
    operations, counted_parameters = profile(profiled, inputs=(image,), verbose=False)
    del profiled
    for _ in range(10):
        model(image)
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(image)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                thop_parameters=int(counted_parameters), thop_raw_operations=int(operations),
                thop_version=getattr(thop, '__version__', None), median_ms=statistics.median(samples),
                mean_ms=statistics.mean(samples), samples_ms=samples)


def run(run_dir, arm, audit_dir, workbook_file, output):
    from ..models.selective_student import build_student

    if output.exists():
        raise FileExistsError(output)
    path, selected_epoch = verify_checkpoint(run_dir, arm, audit_dir)
    state = torch.load(path, map_location='cpu', weights_only=True)
    device = torch.device('cuda:0')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_student(state['config']['student']).to(device).eval()
    model.load_state_dict(state['student'])
    x = torch.randn(1, 3, 256, 256, dtype=torch.float32, device=device,
                    generator=torch.Generator(device=device).manual_seed(42))
    workbook = json.loads(workbook_file.read_text(encoding='utf-8'))
    values = {r['cell']: r['value'] for r in workbook['measurements'] if r['sheet'] == 'Sheet4'}
    before = gpu_state()
    result = measure(model, x)
    after = gpu_state()
    report = dict(training_dataset='UIEB', checkpoint_sha256=file_hash(path), epoch=selected_epoch,
                  device=torch.cuda.get_device_name(device), torch_version=str(torch.__version__),
                  input_shape=list(x.shape), precision='FP32', tf32=False, warmup=10, repetitions=30,
                  timing_scope='Synchronized forward wall time, excluding image I/O and pre/post-processing',
                  gpu_state_before=before, gpu_state_after=after, candidate=result,
                  workbook_sha256=workbook['sha256'],
                  workbook=dict(parameters=values['B11'], thop_raw_operations=values['C11'],
                                median_ms=1000 * values['D11'], cells='Sheet4!B11:D11'))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('arm')
    for field in ('audit_dir', 'workbook_file', 'output'):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
