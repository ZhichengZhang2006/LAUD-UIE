import json
from pathlib import Path

from luad_uie.scripts.score_fusion_uieb import workbook_comparison


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
