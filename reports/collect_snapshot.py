"""Collect existing experiment records; does not train or evaluate models."""
import datetime
import hashlib
import json
from pathlib import Path
import yaml

root = Path('/home/user/luad_uie_prototype_20261031')
dest = root/'reports'/'2026-11-11'
dest.mkdir(parents=True, exist_ok=True)
snapshot = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'root': str(root), 'runs': {}}
tests = {}
for name, eval_name in [('LSUI_teacher_seed42', 'LSUI_teacher_test'), ('E1_color_LSUI_seed42', 'E1_color_LSUI_test'), ('E1_illumination_LSUI_seed42', 'E1_illumination_LSUI_test'), ('E1_detail_LSUI_seed42', 'E1_detail_LSUI_test'), ('E2_LSUI_seed42', 'E2_LSUI_test')]:
    run = root/'runs'/name
    if not run.exists():
        continue
    history = [json.loads(line) for line in (run/'epochs.jsonl').read_text().splitlines() if line.strip()]
    provenance = json.loads((run/'provenance.json').read_text())
    item = {'config': yaml.safe_load((run/'config.yaml').read_text()), 'history': history,
            'best_validation': max(history, key=lambda r: r['val']['psnr']),
            'unique_epochs': len(set(r['epoch'] for r in history)),
            'torch': provenance['torch'], 'manifest_sha256': provenance['manifest_sha256'],
            'split_counts': {k: len(v) for k, v in provenance['splits'].items()},
            'teacher_sha256': provenance.get('teacher_sha256'),
            'checkpoint_files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in run.glob('*.pt')}}
    result_path = root/'runs'/eval_name/'metrics.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        item['test'] = result['summary']
        tests[name] = {r['id']: r for r in result['per_image']}
    snapshot['runs'][name] = item
snapshot['paired_test_comparisons'] = {}
for name, rows in tests.items():
    for reference, refs in tests.items():
        if name == reference or set(rows) != set(refs):
            continue
        deltas = [rows[k]['psnr']-refs[k]['psnr'] for k in rows]
        snapshot['paired_test_comparisons'][name+' vs '+reference] = {
            'images': len(deltas), 'mean_delta_psnr': sum(deltas)/len(deltas),
            'lower_count': sum(d < 0 for d in deltas), 'higher_count': sum(d > 0 for d in deltas)}
snapshot['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'luad_uie').rglob('*.py')}
(dest/'snapshot.json').write_text(json.dumps(snapshot, indent=2))
for name, item in snapshot['runs'].items():
    print(name, 'epochs', item['unique_epochs'], 'best_val', item['best_validation']['val'], 'test', item.get('test'))
print('SAVED', dest/'snapshot.json')
