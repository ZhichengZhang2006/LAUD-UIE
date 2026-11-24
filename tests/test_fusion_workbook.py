import json
from pathlib import Path

import pytest
import torch

from luad_uie.runtime import file_hash
from luad_uie.scripts.score_fusion_uieb import verify_checkpoint, workbook_comparison


def test_workbook_cells_metric_directions_and_missing_references():
    path = Path(__file__).resolve().parents[1] / 'reports/mambacue_workbook_reference.json'
    workbook = json.loads(path.read_text(encoding='utf-8'))
    paired = dict(SSIM=.9245, PSNR=25., UCIQE=.38, E00=4., LPIPS=.12)
    unpaired = dict(UIQM=3.5, UCIQE=.39, URanker=1.8, Entropy=7.5, NIQE=4.2)
    result = workbook_comparison({'UIEB': paired, 'CH60': unpaired, 'U45': unpaired}, workbook)
    by_key = {(r['dataset'], r['metric']):r for r in result}
    assert len(by_key) == 15
    assert by_key['UIEB', 'SSIM']['numerical_win'] is False
    assert by_key['UIEB', 'UCIQE']['workbook_cell'] == 'Sheet3!F28'
    assert by_key['UIEB', 'PSNR']['workbook_cell'] == 'Sheet1!C12'
    assert by_key['U45', 'URanker']['workbook_cell'] == 'Sheet2!I12'
    assert by_key['U45', 'NIQE']['numerical_win'] is False
    assert by_key['CH60', 'NIQE']['numerical_win'] is True
    for metric in ('E00', 'LPIPS'):
        assert by_key['UIEB', metric]['numerical_win'] is None


def test_uieb_eval_requires_the_best_and_final_checkpoints(tmp_path):
    run, audit, arm = tmp_path/'run', tmp_path/'audit', 'free'
    (run/arm).mkdir(parents=True)
    audit.mkdir()
    data = {split:tmp_path/f'{split}.csv' for split in ('train', 'val')}
    for split, manifest in data.items():
        manifest.write_text(f'id,input,target\n{split},a,b\n')
    fingerprints = {split:file_hash(manifest) for split, manifest in data.items()}
    (audit/'initialization.json').write_text(json.dumps({
        'backbone_sha256':'backbone', 'candidate_manifest_sha256':fingerprints,
    }))
    (run/'protocol.json').write_text(json.dumps({
        'parameters':{'epochs':100, 'seed':42, 'batch':1}, 'weights_sha256':'backbone',
        'train_sha256':fingerprints['train'], 'val_sha256':fingerprints['val'],
        'train_ids':['train'], 'val_ids':['val'],
    }))
    (run/'COMPLETE').write_text(arm)
    history = [dict(arm=arm, epoch=epoch, val={'psnr':-abs(epoch-81), 'ssim':.8})
               for epoch in range(1,101)]
    (run/arm/'epochs.jsonl').write_text('\n'.join(map(json.dumps, history)))
    config = {'student':{'architecture':'backbone_fusion'}, 'epochs':100, 'evaluation_resize':256,
              'seed':42, 'batch_size':1, 'data':{key:str(path) for key, path in data.items()}}
    best, last = run/arm/'best.pt', run/arm/'last.pt'
    def write(path, epoch):
        torch.save(dict(**history[epoch-1], config=config, backbone_sha256='backbone'), path)
    write(best, 81)
    write(last, 100)
    assert verify_checkpoint(run, arm, audit) == (best, 81)
    write(best, 100)
    with pytest.raises(ValueError, match='best validation epoch'):
        verify_checkpoint(run, arm, audit)
    write(best, 81)
    write(last, 99)
    with pytest.raises(ValueError, match='final checkpoint'):
        verify_checkpoint(run, arm, audit)
