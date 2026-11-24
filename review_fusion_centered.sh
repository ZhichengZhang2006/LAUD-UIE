#!/bin/sh
set -eu
cd /home/user/luad_fusion_1_0/formal100
pid=$(cat queue.pid)
# The queue creates ordinal only after writing centered's final and best checkpoints.
while [ ! -d standard/ordinal ]; do
    ps -p "$pid" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]' || exit 1
    sleep 60
done
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
exec /home/user/Conda/HVI/bin/python -m luad_uie.scripts.compare_fusion_validation \
    . centered_vs_free_epoch100.json --centered-only --with-sift
