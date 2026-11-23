"""Export a formally validated LSUI fusion checkpoint for DUO detector training."""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import yaml

from ..runtime import file_hash
from .prepare_duo_workbook import CLASSES
from .summarize_fusion_100 import collect


def export_images(model, records, output, device):
    import cv2
    import numpy as np

    for split in ('train', 'test'):
        (output / 'images' / split).mkdir(parents=True, exist_ok=False)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=False)
    exported = []
    with torch.inference_mode():
        for row in records:
            path = Path(row['input'])
            if file_hash(path) != row['sha256']:
                raise ValueError(f'Changed DUO image: {path}')
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Unreadable DUO image: {path}')
            # Direct 256 square resize preserves normalized box coordinates.
            rgb = cv2.cvtColor(cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device).float() / 255
            prediction = model(tensor)
            if prediction.shape != tensor.shape or not torch.isfinite(prediction).all():
                raise ValueError(f'Invalid model output for {path}')
            enhanced = prediction[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            bgr = cv2.cvtColor(np.rint(255 * enhanced).astype(np.uint8), cv2.COLOR_RGB2BGR)
            split, stem = row['split'], f"{row['split']}_{row['id']}"
            image_path = output / 'images' / split / (stem + '.png')
            labels = output / 'labels' / split / (stem + '.txt')
            if not cv2.imwrite(str(image_path), bgr):
                raise OSError(f'Could not write {image_path}')
            text = ''.join(f'{cls} {x:.9f} {y:.9f} {w:.9f} {h:.9f}\n'
                           for cls, x, y, w, h in row['yolo_labels'])
            labels.write_text(text, encoding='ascii')
            exported.append(dict(split=split, id=row['id'], image_sha256=file_hash(image_path),
                                 labels_sha256=file_hash(labels)))
    return exported


def export(run, arm, manifest, output):
    from ..models.selective_student import build_student

    formal = collect(run)
    score = formal['arms'][arm]['best']
    path = run / arm / 'best.pt'
    if file_hash(path) != score['sha256']:
        raise ValueError('Candidate checkpoint changed after formal verification')
    audited = json.loads(manifest.read_text(encoding='utf-8'))
    records = audited['records']
    counts = Counter(row['split'] for row in records)
    if tuple(audited['classes']) != CLASSES or counts != {'train':320, 'test':80}:
        raise ValueError('Expected the audited 320/80 DUO workbook split')
    if (len({(row['split'], row['id']) for row in records}) != len(records)
            or len({row['sha256'] for row in records}) != len(records)):
        raise ValueError('Duplicate DUO sample or image content')
    state = torch.load(path, weights_only=True, map_location='cpu')
    device = torch.device('cuda:0')
    model = build_student(state['config']['student']).to(device).eval()
    model.load_state_dict(state['student'])
    output.mkdir(parents=True, exist_ok=False)
    images = export_images(model, records, output, device)
    config = dict(path=str(output), train='images/train', val='images/train',
                  test='images/test', nc=len(CLASSES), names=list(CLASSES))
    (output / 'dataset.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    metadata = dict(candidate=arm, enhancer_sha256=score['sha256'], selected_epoch=score['epoch'],
                    verified_training_epochs=100, input_size=256, resize='cv2.INTER_AREA',
                    manifest_sha256=file_hash(manifest), images=images)
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('arm', help='Suite and arm, e.g. standard/free')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.run.resolve(), args.arm, args.manifest.resolve(), args.output.resolve())
