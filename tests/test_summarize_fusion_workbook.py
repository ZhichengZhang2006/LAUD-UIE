import json
from pathlib import Path

import pytest

from luad_uie.scripts.score_fusion_uieb import workbook_comparison
from luad_uie.scripts.summarize_fusion_workbook import summarize


def test_quality_gate_keeps_missing_metrics_in_denominator():
    root = Path(__file__).resolve().parents[1]
    workbook = json.loads((root/'reports/mambacue_workbook_reference.json').read_text(encoding='utf-8'))
    lsui_scores = {'SSIM':1., 'PSNR':50., 'UCIQE':.9, 'LPIPS':0., 'E00':0.}
    cells = {'SSIM':'Sheet1!E12', 'PSNR':'Sheet1!F12', 'UCIQE':'Sheet3!K28'}
    original = {f"{row['sheet']}!{row['cell']}":row['value'] for row in workbook['measurements']}
    lsui = [dict(metric=key, candidate=value, mambacue=original.get(cells.get(key)),
                 workbook_cell=cells.get(key), full_source_coverage=True)
            for key, value in lsui_scores.items()]
    config = dict(workbook_sha256=workbook['sha256'], evaluation_size=256)
    paired = dict(**config, checkpoint_sha256='lsui', training_dataset='LSUI',
                  datasets={'LSUI':{'metrics':lsui}})
    good = dict(UIQM=5., UCIQE=.8, URanker=5., Entropy=8., NIQE=1.)
    uieb = dict(**config, checkpoint_sha256='uieb', training_dataset='UIEB',
                image_counts={'CH60':60,'U45':45},
                rows=workbook_comparison({'CH60':good, 'U45':good}, workbook))
    unresolved = summarize(paired, uieb, workbook, {})
    assert unresolved['denominator'] == 10 and unresolved['unverified'] == 10
    assert unresolved['reaches_80_percent'] is None
    evidence = {key:'matching test inventory and metric implementation'
                for key in unresolved['quality_endpoints']}
    outcome = summarize(paired, uieb, workbook, evidence)
    assert outcome['verified_wins'] == 8 and outcome['unverified'] == 2
    assert outcome['reaches_80_percent'] is True
    lsui[0]['candidate'] = .4
    assert summarize(paired, uieb, workbook, evidence)['reaches_80_percent'] is None
    for row in lsui:
        if row['workbook_cell']:
            row['candidate'] = 0.
    assert summarize(paired, uieb, workbook, evidence)['reaches_80_percent'] is False
    lsui[0].update(workbook_cell='Sheet1!B12', mambacue=original['Sheet1!B12'])
    with pytest.raises(ValueError, match='dataset cells'):
        summarize(paired, uieb, workbook, evidence)
