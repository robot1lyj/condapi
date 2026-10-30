#!/bin/bash
# User: pause only H3, preserve perception, restore H3; never start a Pi server.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-earbuds-clean-v2-62000-20261030-r1
cp_root=$root/checkpoints/earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022
cases=$root/test-data/pi05-rtc-earbuds-clean-v2-62000-replay-20261030-r1
image=openpi-pi:thor-trained-rtc-candidate-20260916
jax_image=openpi-pi:thor-trained-rtc-jax-ref-20260917-r3
ref=/artifacts/rtc-earbuds-clean-v2-62000-jax-quantile-7step-20261030-r1
export_dir=/artifacts/rtc-earbuds-clean-v2-62000-onnx-fp32-cache80-7step-20261030-r1
engine=/artifacts/rtc-earbuds-clean-v2-62000-trt-tf32-cache80-7step-20261030-r1
common=(--rm --runtime=nvidia --network host --ipc host -v "$stage:/bench:ro"
  -v "$cp_root:/checkpoints:ro" -v "$root/artifacts:/artifacts" -v "$cases:/cases:ro"
  -v "$root/cache:/cache" -e OPENPI_DATA_HOME=/cache --entrypoint python)
exec 9>"$stage/gpu.lock"
flock -n 9
[[ $(docker inspect pi05-rtc-infer --format '{{.State.Running}}') == false ]]
[[ $(docker inspect yam-agent-perception --format '{{.State.Running}}') == true ]]
/usr/sbin/nvpmodel -q | grep -q 'NV Power Mode: MAXN'
test -f "$cp_root/62000-pytorch-fp32-r1/rtc_manifest.json"
test -f "$cases/cases.json"
[[ ! -d "$root$ref" && ! -d "$root$export_dir" && ! -d "$root$engine" ]]
h3_id=$(docker inspect comfyui-h3 --format '{{.Id}}')
[[ $(docker inspect "$h3_id" --format '{{.State.Running}}') == true ]]
restore_h3() {
  code=$?
  trap - EXIT
  docker start "$h3_id" || { echo H3_RESTORE_FAILED >&2; exit 1; }
  echo "H3_RESTORED_AFTER_GPU_EXIT_$code"
  exit "$code"
}
trap restore_h3 EXIT
docker stop --timeout 30 "$h3_id"
echo PHASE_JAX_REFERENCE
docker run --name pi05-rtc-earbuds62000-jax-20261030 "${common[@]}" \
  -e JAX_PLATFORMS=cuda -e XLA_PYTHON_CLIENT_PREALLOCATE=false "$jax_image" \
  /bench/rtc_jax_reference.py --checkpoint /checkpoints/62000 \
  --training-contract /checkpoints/62000/training_contract.json --cases /cases/cases.json \
  --output "$ref" --num-steps 7
echo PHASE_TORCH_ONNX
docker run --name pi05-rtc-earbuds62000-export-20261030 "${common[@]}" -e JAX_PLATFORMS=cpu "$image" \
  /bench/export_pi05_rtc_onnx.py --checkpoint /checkpoints/62000-pytorch-fp32-r1 \
  --cases /cases/cases.json --jax-reference "$ref" --output "$export_dir" \
  --compute-dtype float32 --num-steps 7 --text-bucket 80 --cache-time-modulation
echo PHASE_TENSORRT_BUILD
docker run --name pi05-rtc-earbuds62000-build-20261030 "${common[@]}" -e JAX_PLATFORMS=cpu "$image" \
  /bench/build_trt_engine.py --source "$export_dir" --output "$engine" --tf32-experiment
echo PHASE_TENSORRT_VALIDATION
docker run --name pi05-rtc-earbuds62000-validate-20261030 "${common[@]}" -e JAX_PLATFORMS=cpu "$image" \
  /bench/validate_pi05_rtc_trt.py --engine "$engine" --checkpoint /checkpoints/62000-pytorch-fp32-r1 \
  --cases /cases/cases.json --jax-reference "$ref" --output "$engine/validation.json" --allow-experimental
python3 - "$root$engine/validation.json" <<'PY'
import json, math, sys
j = json.load(open(sys.argv[1]))
assert j['status'] == 'compared_not_robot_task_validated'
assert j['prefix_use_quantiles'] and j['num_steps'] == 7
assert len(j['cases']) == 9 and all(r['finite'] and r['prefix_exact'] for r in j['cases'])
assert math.isfinite(j['physical_max_abs']) and j['physical_max_abs'] <= 0.005
assert math.isfinite(j['physical_p99_abs']) and j['physical_p99_abs'] <= 0.001
PY
echo PHASE_OFFLINE_GUARD_CHECK
docker run --name pi05-rtc-earbuds62000-guard-20261030 "${common[@]}" -e JAX_PLATFORMS=cpu "$image" \
  /bench/offline_guard_check.py --engine "$engine" --checkpoint /checkpoints/62000-pytorch-fp32-r1 \
  --cases /cases/cases.json --jax-reference "$ref" \
  --output /artifacts/rtc-earbuds-clean-v2-62000-offline-guard-20261030-r1
[[ $(docker inspect pi05-rtc-infer --format '{{.State.Running}}') == false ]]
[[ $(docker inspect yam-agent-perception --format '{{.State.Running}}') == true ]]
echo CONVERTED_VALIDATED_EARBUDS_62000_NO_PI_SERVICE
