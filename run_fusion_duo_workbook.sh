#!/bin/sh
# Run after the verified LSUI enhancer has exported all 400 DUO images.
set -eu
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 EXPORTED_DUO_DIRECTORY OUTPUT_DIRECTORY" >&2
    exit 2
fi
data=$(realpath "$1")
output=$(realpath -m "$2")
root=/home/user/luad_fusion_1_0
python=/home/user/Conda/HVI/bin/python
weights=/home/user/MambaCue-Ablation/UIE-AutoResearch-2026/pretrained/yolov5s.pt
expected=8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee
digest=$(sha256sum "$weights")
test "${digest%% *}" = "$expected"
evaluator=$(sha256sum "$root/eval_yolo_workbook.py")
test "${evaluator%% *}" = bee06b4ac251498d97187ce59d46aec896a7eee16dba45f1bb50ee980119211f
test -s "$data/provenance.json"
test -s "$data/dataset.yaml"
mkdir "$output"
export YOLOv5_AUTOINSTALL=false WANDB_MODE=disabled OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root/duo_local_deps:/home/user/luad_uie_prototype_20261031/stages/scene_stage_20261123/duo_deps"
export MPLCONFIGDIR="$output/mpl_cache" YOLOV5_CONFIG_DIR="$output/yolo_config"

cd "$root/yolov5"
"$python" -u train.py --weights "$weights" --data "$data/dataset.yaml" --epochs 300 \
    --batch-size 16 --imgsz 640 --hyp data/hyps/hyp.scratch-low.yaml --device 0 \
    --workers 2 --seed 0 --patience 0 --noval --project "$output" --name candidate \
    >"$output/train.log" 2>&1
"$python" -c 'import csv,sys; f=open(sys.argv[1],newline=""); epochs=[int(next(iter(r.values()))) for r in csv.DictReader(f)]; assert epochs==list(range(300)), "Expected 300 consecutive epochs"' \
    "$output/candidate/results.csv"
"$python" "$root/eval_yolo_workbook.py" --yolo-root "$root/yolov5" \
    --data "$data/dataset.yaml" --weights "$output/candidate/weights/last.pt" \
    --output "$output/candidate" --task test --batch-size 1 >"$output/evaluate.log" 2>&1
sha256sum "$data/provenance.json" "$data/dataset.yaml" "$output/candidate/weights/last.pt" \
    "$output/candidate/metrics.json" >"$output/sha256.txt"
