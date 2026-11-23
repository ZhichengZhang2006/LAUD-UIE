"""Evaluate frozen enhancements on DUO and geometrically repeatable SIFT keypoints."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

from .benchmark_uie_downstream import DETECT, digest, models, write

DUO = Path('/home/user/UnderWaterDataset/DUO')
DETECTOR_STUDY = Path('/home/user/MambaCue-Ablation/UIE-AutoResearch-2026/round4')
CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']
SIZE = 256
SIFT_SUBSET = 128
SIFT_TRANSFORMS = ((5, .95), (-10, 1.05))
BORDER = 8
TOLERANCE = 3.
CONFIDENCE = .001
NMS_IOU = .6
MAX_DETECTIONS = 300


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def frozen_data(duo, study):
    annotation = duo/'annotations/instances_test.json'
    dataset = load_json(annotation)
    training = study/'data_local/split_manifest.json'
    split = load_json(training)
    test_manifest = load_json(study/'test_once/data_manifest.json')
    identity = test_manifest['identity']
    if (identity['test_annotation_sha256'] != digest(annotation)
            or identity['split_manifest_sha256'] != digest(training)
            or digest(duo/'annotations/instances_train.json') != split['train_annotation_sha256']):
        raise ValueError('DUO annotations or verified split changed')
    categories = sorted(dataset['categories'], key=lambda c: c['id'])
    if [(c['id'], c['name'].lower()) for c in categories] != list(enumerate(CLASS_NAMES, 1)):
        raise ValueError('DUO category mapping changed')
    expected = {row['id']: row for row in test_manifest['records']}
    images = sorted(dataset['images'], key=lambda row: row['id'])
    if (len(expected) != len(images) or len({row['id'] for row in images}) != len(images)
            or len(expected) != identity['image_count']):
        raise ValueError('DUO test set differs from the verified manifest')
    training_hashes = {row['sha256'] for row in split['records']}
    for row in images:
        rec = expected[row['id']]
        filename = row['file_name']
        if (Path(filename).name != filename or filename != rec['source_file']
                or any(row[axis] != rec[axis] for axis in ('width', 'height'))
                or rec['source_sha256'] in training_hashes):
            raise ValueError(f"Unexpected DUO sample or overlap with detector split: {filename}")
    return annotation, images, expected, digest(training)


def detector(study, split_sha):
    weight = study/'runs_local/confirmation/raw2-s202/weights/last.pt'
    record = load_json(study/'results/train_raw2-s202.json')
    identity = record['experiment_identity']
    checkpoint_sha = digest(weight)
    offline_path = study/'code/offline_env.py'
    if (record['status'] != 'TRAINED' or record['mode'] != 'raw2' or record['epochs'] != 15
            or record['test_images_opened'] != 0 or identity['split_manifest_sha256'] != split_sha
            or record['weight_sha256'] != checkpoint_sha or digest(offline_path) != identity['offline_env_sha256']):
        raise ValueError('DUO detector provenance is inconsistent')
    sys.path.insert(0, str(offline_path.parent))
    from offline_env import configure
    runtime = configure()
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression
    network = DetectMultiBackend(str(weight), device=torch.device('cuda:0'), fp16=False).eval()
    names = network.names
    if [names[i] for i in range(len(names))] != CLASS_NAMES:
        raise ValueError('DUO detector class order changed')
    return network, non_max_suppression, checkpoint_sha, runtime['source']


def square_input(path, row, sha):
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha:
        raise ValueError(f'DUO source changed: {path.name}')
    with Image.open(io.BytesIO(blob)) as image:
        if image.size != (row['width'], row['height']):
            raise ValueError(f'Unexpected DUO source dimensions: {path.name}')
        rgb = image.convert('RGB').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.array(rgb)


def tensor_rgb(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32, device='cuda') / 255.


def sift_points(sift, tensor):
    rgb = tensor[0].permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    detected = sift.detect(gray)
    xy = np.array([point.pt for point in detected], dtype=np.float64).reshape(-1, 2)
    return len(detected), np.unique(xy, axis=0)


def warp_points(points, affine):
    return points @ affine[:, :2].T + affine[:, 2]


def inside(points):
    return ((points >= BORDER) & (points < SIZE - BORDER)).all(axis=1)


def repeatability(source, warped, affine):
    projected = warp_points(source, affine)
    left = projected[inside(source) & inside(projected)]
    reverse = warp_points(warped, cv2.invertAffineTransform(affine))
    right = warped[inside(warped) & inside(reverse)]
    eligible = min(len(left), len(right))
    if eligible == 0:
        return {'matches': 0, 'eligible': 0}
    # Each detected position can be counted at most once, even within dense clusters.
    nearby = cKDTree(right).query_ball_point(left, TOLERANCE)
    edges = [(i, j) for i, choices in enumerate(nearby) for j in choices]
    if not edges:
        return {'matches': 0, 'eligible': eligible}
    row, col = np.asarray(edges).T
    graph = csr_matrix((np.ones(len(edges), dtype=np.int8), (row, col)), shape=(len(left), len(right)))
    matched = (maximum_bipartite_matching(graph, perm_type='column') >= 0).sum()
    return {'matches': int(matched), 'eligible': eligible}


def box_rows(prediction, row):
    boxes = []
    for x0, y0, x1, y1, score, label in prediction.cpu().tolist():
        x0, y0, x1, y1 = np.clip((x0, y0, x1, y1), 0, SIZE).tolist()
        if x1 <= x0 or y1 <= y0:
            continue
        sx, sy = row['width'] / SIZE, row['height'] / SIZE
        boxes.append({'image_id': row['id'], 'category_id': int(label) + 1, 'score': score,
                      'bbox': [x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy]})
    return boxes


def coco_scores(annotation, images, predictions):
    # An existing installed wheel is shared with the earlier RUOD pipeline.
    sys.path.append(str(DETECT/'dependencies'))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not predictions:
        return {'ap': 0., 'ap50': 0., 'ap75': 0.}
    gt = COCO(str(annotation))
    gt.dataset.setdefault('info', {})
    evaluator = COCOeval(gt, gt.loadRes(predictions), 'bbox')
    evaluator.params.imgIds = [row['id'] for row in images]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return dict(zip(('ap', 'ap50', 'ap75'), map(float, evaluator.stats[:3])))


def sift_summary(rows):
    pairs = [pair for row in rows for pair in row['transforms']]
    valid = [pair for pair in pairs if pair['eligible']]
    return {'mean_keypoints': float(np.mean([row['keypoints'] for row in rows])),
            'mean_repeatability': float(np.mean([p['matches']/p['eligible'] for p in valid])) if valid else None,
            'mean_repeated_points': float(np.mean([pair['matches'] for pair in pairs])),
            'transform_pairs': len(pairs), 'valid_transform_pairs': len(valid)}


@torch.inference_mode()
def run(checkpoint, output, duo, study, smoke):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    annotation, images, verified, split_sha = frozen_data(duo, study)
    order = sorted(images, key=lambda row: hashlib.sha256(f"luad-sift-v1:{row['id']}".encode()).digest())
    if smoke:
        images = order[:smoke]
    sift_ids = {row['id'] for row in order[:min(SIFT_SUBSET, len(images))]}
    networks, mamba_sha = models(checkpoint)
    yolo, nms, detector_sha, detector_source = detector(study, split_sha)
    sift = cv2.SIFT_create()
    transforms = [cv2.getRotationMatrix2D(((SIZE-1)/2, (SIZE-1)/2), angle, scale)
                  for angle, scale in SIFT_TRANSFORMS]
    output.mkdir(parents=True, exist_ok=False)
    predictions = {arm: [] for arm in networks}
    sift_rows = {arm: [] for arm in networks}
    for index, row in enumerate(images, 1):
        rgb = square_input(duo/'images/test'/row['file_name'], row, verified[row['id']]['source_sha256'])
        image = tensor_rgb(rgb)
        transformed = []
        if row['id'] in sift_ids:
            transformed = [tensor_rgb(cv2.warpAffine(rgb, matrix, (SIZE, SIZE),
                                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
                           for matrix in transforms]
        for label, network in networks.items():
            enhanced = network(image)
            found = nms(yolo(enhanced), conf_thres=CONFIDENCE, iou_thres=NMS_IOU, max_det=MAX_DETECTIONS)[0]
            predictions[label].extend(box_rows(found, row))
            count, points = sift_points(sift, enhanced)
            repeats = []
            for changed, matrix in zip(transformed, transforms):
                _, changed_points = sift_points(sift, network(changed))
                repeats.append(repeatability(points, changed_points, matrix))
            sift_rows[label].append({'id': row['id'], 'keypoints': count, 'transforms': repeats})
        if index % 50 == 0:
            print(f'DUO {index}/{len(images)}', flush=True)
    report = {
        'protocol': {'scope': 'smoke' if smoke else 'full', 'enhancer_size': SIZE, 'detector_size': SIZE,
                     'conf': CONFIDENCE, 'iou': NMS_IOU, 'max_detections': MAX_DETECTIONS,
                     'sift_opencv': cv2.__version__, 'torch': torch.__version__,
                     'sift_rotation_scale': SIFT_TRANSFORMS, 'sift_interior_margin': BORDER,
                     'sift_tolerance_px': TOLERANCE, 'sift_subset_ids': sorted(sift_ids),
                     'annotation_sha256': digest(annotation), 'detector_split_sha256': split_sha,
                     'detector_sha256': detector_sha, 'detector_source': detector_source},
        'checkpoints': {'mambacue': mamba_sha, 'candidate': digest(checkpoint)},
        'images': len(images), 'metrics': {},
    }
    for arm in networks:
        write(output/f'{arm}_coco_boxes.json', predictions[arm])
        write(output/f'{arm}_sift.json', sift_rows[arm])
        report['metrics'][arm] = {'duo': coco_scores(annotation, images, predictions[arm]),
                                  'sift': sift_summary(sift_rows[arm])}
    write(output/'duo_sift.json', report)
    print(json.dumps(report['metrics'], allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--duo', type=Path, default=DUO)
    parser.add_argument('--detector-study', type=Path, default=DETECTOR_STUDY)
    parser.add_argument('--smoke', type=int, default=0, metavar='N')
    args = parser.parse_args()
    if not 0 <= args.smoke <= 16:
        parser.error('--smoke must be in [0, 16]')
    run(args.candidate, args.output, args.duo, args.detector_study, args.smoke)
