#!/bin/bash
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-earbuds-112000-20261021-r1
cp_root=$root/checkpoints/earbuds-pi05-rtc-base-50h-20261012
image=openpi-pi:thor-trained-rtc-candidate-20260916
(cd "$cp_root/112000" && sha256sum -c "$stage/source.sha256")
docker run --rm --name pi05-rtc-earbuds112000-cpu-prepare-20261021 --cpus 8 --memory 80g --network host --ipc host -e JAX_PLATFORMS=cpu -e OMP_NUM_THREADS=8 -v "$stage:/bench:ro" -v "$cp_root:/checkpoints" -v "$root/artifacts:/artifacts" --entrypoint /bin/bash "$image" -c 'python /bench/audit_checkpoint_finite.py --checkpoint /checkpoints/112000 --output /artifacts/rtc-earbuds-112000-audit-20261021-r1.json && python -c '\''import json; assert json.load(open("/artifacts/rtc-earbuds-112000-audit-20261021-r1.json"))["nonfinite_elements"] == 0'\'' && python /bench/prepare_rtc_checkpoint.py --checkpoint /checkpoints/112000 --training-contract /checkpoints/112000/training_contract.json --output /checkpoints/112000-pytorch-fp32-r1'
date --iso-8601=seconds
echo PREPARED_EARBUDS_112000
