#!/bin/bash
# Authorized 2026-10-14: CPU preparation only, preserve the online 200000 model.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-50h-240000-20261014-r1
cp_root=$root/checkpoints/lego-pi05-rtc-base-50h-20260921
image=openpi-pi:thor-trained-rtc-candidate-20260916
(cd "$cp_root/240000" && sha256sum -c "$stage/source.sha256")
docker run --rm --name pi05-rtc-50h240000-cpu-prepare-20261014 --cpus 8 --memory 80g --network host --ipc host -e JAX_PLATFORMS=cpu -e OMP_NUM_THREADS=8 -v "$stage:/bench:ro" -v "$cp_root:/checkpoints" -v "$root/artifacts:/artifacts" --entrypoint /bin/bash "$image" -c 'python /bench/audit_checkpoint_finite.py --checkpoint /checkpoints/240000 --output /artifacts/rtc-50h-240000-audit-20261014-r1.json && python -c '\''import json; assert json.load(open("/artifacts/rtc-50h-240000-audit-20261014-r1.json"))["nonfinite_elements"] == 0'\'' && python /bench/prepare_rtc_checkpoint.py --checkpoint /checkpoints/240000 --training-contract /checkpoints/240000/training_contract.json --output /checkpoints/240000-pytorch-fp32-r1'
docker run --rm --cpus 8 --memory 32g --network host --ipc host -e JAX_PLATFORMS=cpu -e OMP_NUM_THREADS=8 -v "$stage:/bench:ro" -v "$cp_root:/checkpoints:ro" -v "$root/test-data:/data" --entrypoint python "$image" /bench/prepare_rtc_cases.py --source-suite /data/pi05-replay-v1 --source-parquet /data/ABC-130k-two-tasks/lego_sorting/train/data/source-file-000.parquet --checkpoint /checkpoints/240000 --output /data/pi05-rtc-50h-240000-replay-20261014-r1
date --iso-8601=seconds
echo PREPARED_50H_240000_20261014
