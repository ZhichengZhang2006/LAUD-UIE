"""Verify a packaged UIEB backbone and the held-out evaluation inventories."""
import argparse
import csv
import json
from pathlib import Path

import torch

from ..runtime import file_hash
from .prepare_uieb_protocol import prepare


def verify(source_run, source_protocol, weights, datasets, candidate_data, output):
    state = torch.load(weights, weights_only=True, map_location='cpu')
    original = source_run / 'checkpoint_ema.pt'
    full = torch.load(original, weights_only=True, map_location='cpu')
    if not state or set(state) != {k for k in full if k.startswith(('encoder.', 'decoder.dec'))}:
        raise ValueError('The archive must include the full four-stage encoder and three decoder blocks')
    if any(full[k].dtype != t.dtype or not torch.equal(full[k], t) for k, t in state.items()):
        raise ValueError('The packaged tensors differ from the source EMA checkpoint')
    source = json.loads((source_run / 'config.json').read_text())
    if source['dataset'] != 'UIEB' or source['mode'] != 'train':
        raise ValueError('Expected a UIEB training record')
    for split, digest in source['split_hashes'].items():
        if file_hash(source_protocol / f'UIEB_{split}.json') != digest:
            raise ValueError(f'Changed pretraining {split} manifest')

    # Re-read image bytes and RGB pixels under the exact source protocol.
    prepare(source_protocol, datasets, output)
    audit = json.loads((output / 'audit.json').read_text())
    if any(audit['excluded'].values()):
        raise ValueError('The original split includes images excluded by the held-out overlap audit')
    for split in ('train', 'val', 'test'):
        with (candidate_data / f'{split}.csv').open(newline='', encoding='utf-8-sig') as f:
            actual = list(csv.DictReader(f))
        with (output / f'{split}.csv').open(newline='', encoding='utf-8-sig') as f:
            expected = list(csv.DictReader(f))
        if actual != expected:
            raise ValueError(f'Candidate {split} images or order differ from the original protocol')
    result = dict(source_run=source['id'], epochs_budget=source['epochs'],
                  original_ema_sha256=file_hash(original), backbone_sha256=file_hash(weights),
                  identical_tensors=len(state), split_sha256=source['split_hashes'],
                  candidate_manifest_sha256={k:file_hash(candidate_data / f'{k}.csv') for k in audit['sizes']},
                  sizes=audit['sizes'], exclusions=audit['excluded'],
                  no_reference_sizes={k:len(v) for k,v in audit['no_reference_fingerprints'].items()})
    (output / 'initialization.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source_run', 'source_protocol', 'weights', 'datasets', 'candidate_data', 'output'):
        parser.add_argument(name, type=Path)
    verify(**vars(parser.parse_args()))
