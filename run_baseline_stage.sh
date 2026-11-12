#!/bin/bash
set -euo pipefail
cd /home/user/luad_uie_prototype_20261031/stages/20261112_baseline
exec 9>runs/baseline_stage.lock
flock -n 9 || exit 0
trap 'code=$?; printf "exit_code=%s\n" "$code" > runs/baseline_stage.exit; exit "$code"' EXIT
PY=/home/user/Conda/CDF_FFT/bin/python
$PY -m luad_uie.scripts.safe_train --config configs/E0_LSUI_student.yaml --output runs/E0_LSUI_seed42
$PY -m luad_uie.scripts.evaluate --checkpoint runs/E0_LSUI_seed42/best.pt --manifest data/LSUI/test.csv --output runs/E0_LSUI_test
for name in E1_color E1_illumination E1_detail E2; do
 $PY -m luad_uie.scripts.evaluate --checkpoint /home/user/luad_uie_prototype_20261031/runs/${name}_LSUI_seed42/best.pt --manifest data/LSUI/test.csv --output runs/${name}_corrected_test --baseline runs/E0_LSUI_test/metrics.json
done
printf "BASELINE_STAGE_COMPLETE\n" > runs/baseline_stage.complete
