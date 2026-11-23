#!/bin/sh
set -eu
cd /home/user/luad_fusion_1_0/formal100
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
data=/home/user/luad_uie_prototype_20261031/stages/cf_clean_20261119/data/CF_LSUI
train() {
    /home/user/Conda/HVI/bin/python -m luad_uie.scripts.train_backbone_fusion \
        "$data/train.csv" "$data/val.csv" ../weights/mambacue_backbone_uieb_ema.pth "$@" \
        --epochs 100 --batch 1 --train-limit 512 --val-limit 128 --seed 42
}
train standard --arms free centered ordinal fixed utility
train fitted --arms free centered --head-fit-images 64
