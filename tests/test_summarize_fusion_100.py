import json

import pytest
import torch

from luad_uie.runtime import file_hash
from luad_uie.scripts.summarize_fusion_100 import BACKBONE_SHA256, SUITES, collect, epochs_for_arm


def test_incomplete_suite_has_no_ranking(tmp_path):
    with pytest.raises(ValueError, match='awaits completion'):
        collect(tmp_path)


def test_missing_epoch_is_detected_even_with_100_rows(tmp_path):
    (tmp_path / 'epochs.jsonl').write_text('\n'.join(json.dumps({'epoch': i}) for i in [1] + list(range(1, 100))))
    with pytest.raises(ValueError, match='consecutive'):
        epochs_for_arm(tmp_path)


def test_all_seven_runs_and_paired_selection(tmp_path):
    train = [f'train{i}' for i in range(512)]
    for suite, arms in SUITES.items():
        path = tmp_path / suite
        path.mkdir()
        fit = 64 if suite == 'fitted' else 0
        protocol = dict(parameters=dict(epochs=100, batch=1, seed=42, train_limit=512, val_limit=128,
                                        head_fit_images=fit, arms=list(arms), smoke_test=False),
                        weights_sha256=BACKBONE_SHA256, train_ids=train, val_ids=[f'v{i}' for i in range(128)],
                        train_sha256='train', val_sha256='val', head_fit_ids=train[:fit])
        (path / 'protocol.json').write_text(json.dumps(protocol))
        (path / 'COMPLETE').write_text(' '.join(arms))
        for arm in arms:
            folder = path / arm
            folder.mkdir()
            rows = [dict(arm=arm, epoch=i, val=dict(psnr=1 if i in (14, 81) else 0, ssim=i / 100))
                    for i in range(1, 101)]
            (folder / 'epochs.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            for name, index in [('last', -1), ('best', 13)]:
                state = dict(**rows[index], backbone_sha256=BACKBONE_SHA256,
                             config={'epochs':100, 'evaluation_resize':256})
                torch.save(state, folder / (name + '.pt'))
            if arm in ('fixed', 'utility'):
                (folder / 'teacher.sha256').write_text(file_hash(path / 'free/best.pt'))
    report = collect(tmp_path)
    assert len(report['arms']) == 7
    for scores in report['arms'].values():
        assert scores['best']['epoch'] == 14 and scores['best']['ssim'] == .14
        assert scores['last']['epoch'] == 100
    (tmp_path / 'standard/fixed/teacher.sha256').write_text('other')
    with pytest.raises(ValueError, match='wrong distillation teacher'):
        collect(tmp_path)
