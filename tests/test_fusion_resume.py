import json

import torch
from torch.utils.data import DataLoader

from luad_uie.scripts.train_backbone_fusion import recover_epoch, restore_training_rng


def test_legacy_resume_replays_train_loader_order():
    reference = torch.Generator().manual_seed(42)
    loader = DataLoader(list(range(19)), batch_size=2, shuffle=True, generator=reference)
    for epoch in range(3):
        list(loader)
    expected = list(loader)
    recovered = torch.Generator().manual_seed(42)
    resumed = DataLoader(list(range(19)), batch_size=2, shuffle=True, generator=recovered)
    assert restore_training_rng({'epoch':3}, resumed, recovered) == 'legacy_loader_replay'
    assert [row.tolist() for row in resumed] == [row.tolist() for row in expected]


def test_last_durable_epoch_repairs_best_and_archives_extra_log(tmp_path):
    rows = [dict(arm='free', epoch=i, val={'psnr':float(i)}) for i in (1,2)]
    log = tmp_path / 'epochs.jsonl'
    log.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    torch.save(rows[0], tmp_path / 'last.pt')
    latest, best = recover_epoch(tmp_path, 'free', 100)
    assert latest == rows[0] and best == 1.
    assert torch.load(tmp_path/'best.pt', weights_only=True) == rows[0]
    assert json.loads(log.read_text()) == rows[0]
    assert json.loads((tmp_path/'discarded_epoch.jsonl').read_text()) == rows[1]
