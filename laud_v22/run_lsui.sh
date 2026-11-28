#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
python=/home/user/Conda/HVI/bin/python
export PYTHONPATH="$root:$root/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_HOME="$root/torch_cache"

exec 9>"$root/lsui.lock"
flock -n 9
mkdir -p "$root/runs/LSUI_ablation" "$TORCH_HOME/hub/checkpoints"
if ! "$python" -c 'import lpips'; then
    "$python" -m pip install --no-deps --target "$root/deps" 'lpips==0.1.4'
fi
if [[ ! -f "$TORCH_HOME/hub/checkpoints/vgg16-397923af.pth" ]]; then
    for directory in /home/user/pre_trained_VGG /home/user/luad_fusion_1_0 /root/.cache/torch /home/user/.cache/torch; do
        [[ -d "$directory" ]] || continue
        weights=$(find "$directory" -maxdepth 6 -type f -name 'vgg16-397923af.pth' -print -quit)
        if [[ -n "$weights" ]]; then
            ln -s "$weights" "$TORCH_HOME/hub/checkpoints/vgg16-397923af.pth"
            break
        fi
    done
fi

if [[ ! -f configs/LSUI.json ]]; then
    "$python" laud_v22/freeze_lsui.py --output "$root/manifests/LSUI"
    "$python" laud_v22/make_configs.py --data-root "$root/manifests" \
        --runs-root "$root/runs" --metric-repo /home/user/MambaCue --output "$root/configs"
fi
for arm in M0 AFF A AB AC ABC; do
    run="$root/runs/LSUI_ablation/${arm}_s42"
    if [[ -f "$run/COMPLETE" && $(cat "$run/COMPLETE") == 300 ]]; then continue; fi
    resume=()
    if [[ -f "$run/last.pt" ]]; then resume=(--resume); fi
    "$python" laud_v22/train_v22.py --config "$root/configs/LSUI.json" \
        --arm "$arm" --epochs 300 --seed 42 --output "$run" "${resume[@]}" \
        2>&1 | tee "$root/runs/LSUI_ablation/${arm}_s42.log"
done
