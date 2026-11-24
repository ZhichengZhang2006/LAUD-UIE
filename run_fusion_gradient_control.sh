#!/bin/sh
# Run on school-gpu after the existing UIEB job and its scorer release GPU memory.
set -eu
root=/home/user/luad_fusion_1_0
cd "$root/gradient_control_stage"
for name in uieb_free_200 score_uieb; do
    pid=$(cat "$root/$name.pid")
    while ps -p "$pid" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; do
        sleep 60
    done
done
test -f "$root/uieb_free_200/COMPLETE"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
data=/home/user/luad_uie_prototype_20261031/stages/cf_clean_20261119/data/CF_LSUI
exec /home/user/Conda/HVI/bin/python -u -m luad_uie.scripts.train_backbone_fusion \
    "$data/train.csv" "$data/val.csv" ../weights/mambacue_backbone_uieb_ema.pth \
    control --arms gradient --epochs 100 --train-limit 512 --val-limit 128 --batch 1 --seed 42
