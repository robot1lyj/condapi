#!/bin/bash
# CPU-only preparation. No GPU runtime and no online inference service.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-earbuds-clean-v2-62000-20261030-r1
cp_root=$root/checkpoints/earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022
image=openpi-pi:thor-trained-rtc-candidate-20260916
exec 9>"$stage/prepare.lock"
flock -n 9
(cd "$cp_root/62000" && sha256sum -c "$stage/source.sha256")
docker run --rm --name pi05-rtc-earbuds62000-cpu-prepare-20261030 \
  --cpus 8 --memory 80g --network host --ipc host \
  -e JAX_PLATFORMS=cpu -e OMP_NUM_THREADS=8 \
  -v "$stage:/bench:ro" -v "$cp_root:/checkpoints" \
  -v "$root/artifacts:/artifacts" --entrypoint /bin/bash "$image" -c '
python /bench/audit_checkpoint_finite.py --checkpoint /checkpoints/62000 --output /artifacts/rtc-earbuds-clean-v2-62000-audit-20261030-r1.json &&
python -c '\''import json; assert json.load(open("/artifacts/rtc-earbuds-clean-v2-62000-audit-20261030-r1.json"))["nonfinite_elements"] == 0'\'' &&
python /bench/prepare_rtc_checkpoint.py --checkpoint /checkpoints/62000 --training-contract /checkpoints/62000/training_contract.json --output /checkpoints/62000-pytorch-fp32-r1'
python3 "$stage/rebind_cases.py"
date --iso-8601=seconds
echo CPU_PREPARED_EARBUDS_CLEAN_V2_62000_NO_SERVICE
