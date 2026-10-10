#!/usr/bin/env bash
# Frozen one-shot Thor benchmark launcher. Run with sudo on the Thor host.
set -euo pipefail
stage=/home/wuyan-lyj/thor/xr1/rtc-20261010-r1
exec python3 /home/wuyan-lyj/thor/pi/probes/rtc-50h-200000-20261010-r1/maxn_session.py -- \
  docker run --rm --name xr1-rtc-benchmark-20261010-r1 --runtime nvidia --gpus all --ipc host \
  --entrypoint python -e HF_HOME=/hf -e PYTHONPATH=/source -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$stage:/bench" -v "$stage/source:/source:ro" -v "$stage/hf-home:/hf:ro" \
  -v /home/wuyan-lyj/thor/xr1/checkpoints/base-5b-ee21d524:/checkpoint:ro \
  -v /home/wuyan-lyj/thor/pi/test-data/pi05-replay-v1:/cases:ro \
  xiaomi-xr1:thor-native-rtc-20261010 /bench/benchmark_xr1_rtc.py \
  --checkpoint /checkpoint/model_states.pt --fixtures /cases --output /bench/results.json \
  --prefixes 0 1 3 6 8 10 --sizes 224 384 --warmups 3 --repeats 30
