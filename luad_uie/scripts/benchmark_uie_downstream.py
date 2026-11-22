"""Compare the same LSUI enhancers through fixed SUIM and RUOD task pipelines."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from luad_uie.models.selective_student import build_student

MAMBA = Path('/home/user/MambaCue')
STUDY = Path('/home/user/physician_curve_study_20261122')
DETECT = Path('/home/user/physician_detection_v30_20261028')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def models(checkpoint):
    sys.path.insert(0, str(MAMBA))
    from model import myModel
    known = json.loads((MAMBA/'supplement_20261112/protocol/historical_checkpoints.json').read_text())['LSUI']
    if digest(known['path']) != known['sha256']:
        raise ValueError('Unexpected MambaCue checkpoint')
    baseline = myModel()
    baseline.load_state_dict(torch.load(known['path'], map_location='cpu', weights_only=True))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    candidate = build_student(saved['config']['student'])
    candidate.load_state_dict(saved['student'])
    return {'raw': torch.nn.Identity().eval().cuda(), 'mambacue': baseline.eval().cuda(),
            'candidate': candidate.eval().cuda()}, known['sha256']


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


@torch.inference_mode()
def run(output, checkpoint):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    networks, baseline_sha = models(checkpoint)
    sys.path.insert(0, str(STUDY))
    from calibrate_boundaries import images, measure, summarize
    calibration, test = [STUDY/f'suim_{name}.zip' for name in ('calibration120','test110')]
    cal_rows, test_rows = [list(images(p)) for p in (calibration, test)]
    if {r[0] for r in cal_rows} & {r[0] for r in test_rows}:
        raise ValueError('SUIM overlap')
    output.mkdir(parents=True, exist_ok=False)
    provenance = {'mambacue': baseline_sha, 'candidate': digest(checkpoint)}
    thresholds = (25, 50, 75, 100, 125, 150)
    raw_cal = measure(lambda x:x, cal_rows, thresholds)
    threshold = max(thresholds, key=lambda t:summarize([r for r in raw_cal if r['high']==t])['f1'])
    boundary = {}
    for label, net in networks.items():
        def transform(rgb):
            tensor = torch.from_numpy(rgb).permute(2,0,1)[None].cuda()
            return net(tensor)[0].permute(1,2,0).cpu().numpy()
        rows = measure(transform, test_rows, (threshold,))
        boundary[label] = summarize(rows)
    write(output/'suim.json', {'threshold':threshold, 'test_images':len(test_rows),
        'archives':{p.name:digest(p) for p in (calibration,test)}, 'metrics':boundary, **provenance})
    print('SUIM',boundary,flush=True)
    sys.path.insert(0, str(DETECT/'Physician-for-UIE'))
    sys.path.append(str(DETECT/'dependencies'))
    from detection30.data import CocoBoxes
    from detection30.detector import build_detector
    from detection30.evaluate import evaluate
    cfg = json.loads((DETECT/'runs/configs/seed42_raw_control.json').read_text())
    sets = [DETECT/'data'/f'{name}.json' for name in ('train','val','test')]
    ids = [{im['file_name'] for im in json.loads(p.read_text())['images']} for p in sets]
    if ids[2] & (ids[0] | ids[1]):
        raise ValueError('RUOD split overlaps')
    dataset = CocoBoxes(sets[2], cfg['images'])
    detector_file = Path(cfg['output'])/'best.pt'
    state = torch.load(detector_file,map_location='cpu',weights_only=True)
    if state['categories'] != dataset.categories or state.get('enhancer_sha256') is not None:
        raise ValueError('Unexpected detector labels or training preprocessing')
    detector = build_detector(len(dataset.categories)).eval().cuda()
    detector.load_state_dict(state['model'])
    for label, net in networks.items():
        folder = output/label
        folder.mkdir()
        result = evaluate(detector, dataset, net, torch.device('cuda'), batch_size=1, workers=0, output=folder)
        write(folder/'provenance.json', {'checkpoint':provenance.get(label), 'detector':digest(detector_file),
                                         'annotations':digest(sets[2])})
        print('RUOD',label,result['coco']['AP'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.candidate.resolve())
