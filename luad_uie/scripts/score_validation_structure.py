"""Score SIFT stability on paired development images without opening DUO test."""
import argparse
from pathlib import Path

import cv2
import torch

from ..datasets import PairedDataset, read_manifest
from ..models.selective_student import build_student
from .benchmark_duo_sift import SIZE, SIFT_TRANSFORMS, digest, repeatability, sift_points, sift_summary, tensor_rgb, write
from .train_matched_quality import subset


@torch.inference_mode()
def run(manifest, output, checkpoints, limit, seed):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = subset(read_manifest(manifest), limit, seed)
    networks = {}
    weights = {}
    for path in checkpoints:
        state = torch.load(path, map_location='cpu', weights_only=True)
        arm = state['arm']
        if arm in networks:
            raise ValueError(f'Duplicate checkpoint arm: {arm}')
        network = build_student(state['config']['student'])
        network.load_state_dict(state['student'])
        networks[arm] = network.eval().cuda()
        weights[arm] = digest(path)

    sift = cv2.SIFT_create()
    transforms = [cv2.getRotationMatrix2D(((SIZE-1)/2, (SIZE-1)/2), angle, scale)
                  for angle, scale in SIFT_TRANSFORMS]
    measured = {arm: [] for arm in networks}
    for sample in PairedDataset(rows, resize=SIZE):
        image = sample['input'].unsqueeze(0).cuda()
        rgb = sample['input'].permute(1, 2, 0).mul(255).round().to(torch.uint8).contiguous().numpy()
        warped = [tensor_rgb(cv2.warpAffine(rgb, matrix, (SIZE, SIZE),
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))
                  for matrix in transforms]
        for arm, network in networks.items():
            count, points = sift_points(sift, network(image))
            repeat = [repeatability(points, sift_points(sift, network(view))[1], matrix)
                      for view, matrix in zip(warped, transforms)]
            measured[arm].append({'id': sample['id'], 'keypoints': count, 'transforms': repeat})
    write(output, {'manifest_sha256': digest(manifest), 'image_size': SIZE, 'sift_opencv': cv2.__version__,
                   'transforms': SIFT_TRANSFORMS, 'ids': [row['id'] for row in rows], 'checkpoints': weights,
                   'summary': {arm: sift_summary(results) for arm, results in measured.items()},
                   'per_image': measured})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    run(args.manifest, args.output, args.checkpoints, args.limit, args.seed)
