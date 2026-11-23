#!/bin/sh
set -eu
cd /home/user/luad_fusion_1_0/formal100
pid=$(cat queue.pid)
while [ ! -f fitted/COMPLETE ]; do
    test ! -f standard/STOPPED_OOM && test ! -f fitted/STOPPED_OOM || exit 42
    ps -p "$pid" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]' || exit 1
    sleep 60
done
exec /home/user/Conda/HVI/bin/python -m luad_uie.scripts.summarize_fusion_100 . formal_summary.json
