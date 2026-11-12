import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_config(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config['experiment'] not in ('E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'RG-full', 'RG-scale'):
        raise ValueError('Unknown experiment')
    if config.get('clip', False):
        raise NotImplementedError('CLIP is intentionally excluded from MVP')
    if config['experiment'] != 'E0' and config['teacher']['mode'] == 'none':
        raise ValueError('Distillation requires a checkpoint teacher or explicit oracle debugging')
    if config['experiment'] == 'E1' and config.get('view') not in ('color', 'illumination', 'detail'):
        raise ValueError('E1 requires view=color/illumination/detail')
    if config['experiment'] in ('E4', 'E5') and not all(config['data'].get(k) for k in ('utility_train', 'utility_eval')):
        raise ValueError('E4/E5 require disjoint training-internal utility_train and utility_eval manifests')
    return config


def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
