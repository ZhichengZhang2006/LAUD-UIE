#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs
exec 9>runs/router.lock
flock -n 9 || exit 0
trap 'code=$?; echo "$code" > runs/router.exit' EXIT
PY=/home/user/Conda/CDF_FFT/bin/python
B=/home/user/luad_uie_prototype_20261031/stages/20261112_baseline/runs/E0_LSUI_test/metrics.json
for name in RG-full RG-scale; do
  "$PY" -m luad_uie.scripts.safe_train --config "configs/${name}_LSUI.yaml" --output "runs/${name}_seed42"
done
for name in RG-full RG-scale; do
  "$PY" -m luad_uie.scripts.evaluate --checkpoint "runs/${name}_seed42/best.pt" --manifest /home/user/luad_uie_prototype_20261031/data/LSUI/test.csv --output "runs/${name}_test" --baseline "$B"
done
