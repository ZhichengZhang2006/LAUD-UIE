"""Stop on CUDA OOM, leaving a machine-readable record and nonzero exit."""
import argparse
import json
from pathlib import Path
import traceback
import torch
from ..runtime import load_config
from ..trainers.engine import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume')
    args = p.parse_args()
    config = load_config(args.config)
    try:
        train(config, args.output, args.resume)
    except RuntimeError as error:
        if isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower():
            path = Path(args.output)
            path.mkdir(parents=True, exist_ok=True)
            (path/'STOPPED_OOM.json').write_text(json.dumps({
                'status': 'stopped_oom', 'error': str(error), 'config': config,
                'retry': False, 'traceback': traceback.format_exc()}, indent=2), encoding='utf-8')
            raise SystemExit(42)
        raise


if __name__ == '__main__':
    main()
