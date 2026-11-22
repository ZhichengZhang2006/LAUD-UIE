"""Collect per-image MambaCue reference measurements using the supplied LSUI weights."""
import hashlib
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rgb(path):
    with Image.open(path) as img:
        rgb = img.convert('RGB').resize((256, 256), Image.BILINEAR)
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1)[None].cuda().float() / 255


@torch.inference_mode()
def main(output, student_checkpoint=None):
    repo = Path('/home/user/MambaCue')
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from utils import metrics
    from utils.uranker.uranker_utils import build_model, get_option
    protocol = repo/'supplement_20261112/protocol'
    if student_checkpoint is None:
        from model import myModel
        checkpoint = json.loads((protocol/'historical_checkpoints.json').read_text())['LSUI']
        if digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Baseline checkpoint SHA mismatch')
        model = myModel()
        weights = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
        model_name = 'MambaCue'
    else:
        from luad_uie.models.selective_student import build_student
        checkpoint = {'path': str(student_checkpoint), 'sha256': digest(student_checkpoint)}
        state = torch.load(student_checkpoint, map_location='cpu', weights_only=True)
        model = build_student(state['config']['student'])
        weights = state['student']
        model_name = 'LUAD-UIE'
    model.load_state_dict(weights, strict=True)
    model.cuda().eval()
    manifest = json.loads((protocol/'LSUI_test.json').read_text())
    images = {'LSUI': manifest}
    for name in ('U45', 'CH60'):
        paths = sorted((Path('/home/user/UnderWaterDataset')/name/'test').iterdir())
        images[name] = [{'input': str(path), 'name': path.name} for path in paths
                        if path.suffix.lower() in ('.jpg', '.png', '.bmp', '.jpeg')]
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'model':model_name, 'checkpoint':checkpoint, 'model_parameters':sum(p.numel() for p in model.parameters()),
                'device':torch.cuda.get_device_name(), 'torch':str(torch.__version__),
                'preprocessing':'PIL RGB bilinear 256x256 float32 /255',
                'source_sha256':{f:digest(repo/f) for f in ('model.py','encoder.py','output_heads.py','prior.py','utils/metrics.py','utils/niqe_utils.py')},
                'manifest_sha256':digest(protocol/'LSUI_test.json')}
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2))
    example = load_rgb(manifest[0]['input'])
    for _ in range(10):
        model(example)
    torch.cuda.synchronize()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(30):
        start = time.perf_counter()
        model(example)
        torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-start))
    (output/'profile.json').write_text(json.dumps({'median_ms':np.median(times),'p90_ms':np.percentile(times,90),
        'peak_bytes':torch.cuda.max_memory_allocated(),'shared_gpu':True},indent=2))
    ranker = None
    for dataset, rows in images.items():
        if not rows:
            raise ValueError(dataset)
        folder = output/dataset
        folder.mkdir()
        if dataset != 'LSUI' and ranker is None:
            options = get_option('utils/uranker/URanker.yaml')['model']
            options.update(cuda=False,resume_ckpt_path='utils/uranker/URanker_ckpt.pth')
            ranker = build_model(options).cpu().eval()
        scores = []
        with (folder/'metrics.jsonl').open('x') as stream:
            for i,row in enumerate(rows):
                inp = load_rgb(row['input'])
                pred = model(inp).clamp(0,1).permute(0,2,3,1).cpu().numpy()
                result = {'UCIQE':float(metrics.getUCIQE(pred))}
                if 'target' in row:
                    if digest(row['target']) != row['target_sha256'] or digest(row['input']) != row['input_sha256']:
                        raise ValueError('Dataset content changed: '+row['name'])
                    target = load_rgb(row['target']).permute(0,2,3,1).cpu().numpy()
                    result.update(SSIM=float(metrics.getSSIM(pred,target,1.)), PSNR=float(metrics.getPSNR(pred,target,1.)),
                                  E00=float(deltaE_ciede2000(rgb2lab(pred[0]),rgb2lab(target[0])).mean()))
                    np.save(folder/(str(i)+'.npy'),pred[0],allow_pickle=False)
                else:
                    result.update(UIQM=float(metrics.getUIQM(pred)), Entropy=float(metrics.getEntropy(pred)),
                                  NIQE=float(metrics.getNIQE(pred)), URanker=float(metrics.getURanker(pred,ranker)))
                stream.write(json.dumps({'id':row['name'],**result},allow_nan=False)+'\n')
                stream.flush()
                scores.append(result)
        summary = {k:float(np.mean([r[k] for r in scores])) for k in scores[0]}
        (folder/'summary.json').write_text(json.dumps({'count':len(rows),**summary},indent=2))
        print(dataset,len(rows),summary,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--student-checkpoint', type=Path)
    args = parser.parse_args()
    main(args.output.resolve(), args.student_checkpoint.resolve() if args.student_checkpoint else None)
