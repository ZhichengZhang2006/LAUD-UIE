"""Put the verified torchvision VGG-16 weights in a local PyTorch cache."""
import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

FILENAME = 'vgg16-397923af.pth'
SHA256 = '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'


def validate(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError(f'Checksum mismatch for VGG-16: {path}')


def stage(source, torch_home):
    target = torch_home / 'hub' / 'checkpoints' / FILENAME
    if target.exists():
        validate(target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish the cache file only after verifying the exact bytes PyTorch will read.
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='vgg16-', suffix='.part', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open('rb') as weights:
                shutil.copyfileobj(weights, stream, length=8 * 1024 * 1024)
            stream.close()
            validate(temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Path to vgg16-397923af.pth')
    parser.add_argument('torch_home', type=Path, help='Value of TORCH_HOME for the evaluation process')
    args = parser.parse_args()
    print(stage(args.source, args.torch_home).resolve())
