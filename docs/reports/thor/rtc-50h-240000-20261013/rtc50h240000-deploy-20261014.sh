#!/bin/bash
# Authorized 2026-10-14. Freeze before execution; never overwrite a running file.
# Keep 200000/230000 and PLD intact. Restore 200000 if any GPU/online gate fails.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-50h-240000-20261014-r1
cp_root=$root/checkpoints/lego-pi05-rtc-base-50h-20260921
tag=rtc-50h-240000
suffix=20261014-r1
image=openpi-pi:thor-trained-rtc-candidate-20260916
jax_image=openpi-pi:thor-trained-rtc-jax-ref-20260917-r3
cases=$root/test-data/pi05-rtc-50h-240000-replay-20261014-r1
ref=/artifacts/$tag-jax-quantile-7step-$suffix
export_dir=/artifacts/$tag-onnx-fp32-cache80-7step-$suffix
engine=/artifacts/$tag-trt-tf32-cache80-7step-$suffix
ckpt=/checkpoints/240000-pytorch-fp32-r1
candidate=pi05-rtc-50h240000-candidate-20261014
preserved=pi05-rtc-50h200000-preserved-20261014
common=(--rm --runtime=nvidia --network host --ipc host -v "$stage:/bench:ro" -v "$cp_root:/checkpoints:ro" -v "$root/artifacts:/artifacts" -v "$cases:/cases:ro" -v "$root/cache:/cache" -e OPENPI_DATA_HOME=/cache --entrypoint python)
ready() {
  for tick in {1..90}; do
    [[ $(docker inspect "$1" --format '{{.State.Running}}') == true ]] || { docker logs --tail 20 "$1"; return 1; }
    if curl -fsS --max-time 2 http://192.168.250.1:8000/healthz >/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}
normal_power() {
  for tick in {1..25}; do
    nvpmodel -q | grep -q 'NV Power Mode: 120W' && return 0
    sleep 2
  done
  echo 'Previous MAXN session has not exited; refusing to nest sessions' >&2
  return 1
}
start_online() {
  systemd-run --unit="$1" /usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash -c "docker start pi05-rtc-infer && exec /usr/bin/python3 $root/probes/rtc-step-sweep-20260917/watch_docker_container.py --container pi05-rtc-infer" || return 1
  ready pi05-rtc-infer
}
restore_200000() {
  normal_power || return 1
  start_online thor-pi-maxn-50h200000-restored-240000-20261014 || return 1
  docker exec pi05-rtc-infer python /bench/smoke_pi05_rtc_ws.py --url ws://192.168.250.1:8000 --cases /cases/cases.json --expected-weights-sha256 298714be810f5e7e822bcac55c5efe55789f1c96c0ac4ed0a1caa86f2670e42a --expected-prefix-norm quantile --output /tmp/200000-restored-240000-failure.json || return 1
  docker cp pi05-rtc-infer:/tmp/200000-restored-240000-failure.json "$stage/200000-restored-240000-failure.json"
}
if [[ ${1:-} == --gpu ]]; then
  echo PHASE_JAX_REFERENCE
  docker run "${common[@]}" -e JAX_PLATFORMS=cuda -e XLA_PYTHON_CLIENT_PREALLOCATE=false "$jax_image" /bench/rtc_jax_reference.py --checkpoint /checkpoints/240000 --training-contract /checkpoints/240000/training_contract.json --cases /cases/cases.json --output "$ref" --num-steps 7
  echo PHASE_TORCH_ONNX
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/export_pi05_rtc_onnx.py --checkpoint "$ckpt" --cases /cases/cases.json --jax-reference "$ref" --output "$export_dir" --compute-dtype float32 --num-steps 7 --text-bucket 80 --cache-time-modulation
  echo PHASE_TENSORRT_BUILD
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/build_trt_engine.py --source "$export_dir" --output "$engine" --tf32-experiment
  echo PHASE_TENSORRT_VALIDATION
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/validate_pi05_rtc_trt.py --engine "$engine" --checkpoint "$ckpt" --cases /cases/cases.json --jax-reference "$ref" --output "$engine/validation.json" --allow-experimental
  echo PHASE_CANDIDATE_PROTOCOL
  docker create --name "$candidate" --restart no --runtime=nvidia --network host --ipc host -e OPENPI_DATA_HOME=/cache -e JAX_PLATFORMS=cpu -v "$stage:/bench:ro" -v "$root/artifacts:/artifacts:ro" -v "$cp_root/240000-pytorch-fp32-r1:/ckpt:ro" -v "$cases:/cases:ro" -v "$root/cache:/cache" --entrypoint python "$image" /bench/serve_pi05_rtc_trt.py --engine "$engine" --checkpoint /ckpt --norm /ckpt/assets/yam/norm_stats.json --warmup-sample /cases/episode-000095-early.npz --warmup-rtc /bench/rtc_warmup_d0.json --host 192.168.250.1 --port 8000 --allow-validated-tf32-7step --validation "$engine/validation.json" --jax-reference "$ref"
  trap 'docker stop "$candidate" >/dev/null 2>&1 || true' EXIT
  docker start "$candidate"
  ready "$candidate"
  weights_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_weights_sha256"])' "$cp_root/240000-pytorch-fp32-r1/rtc_manifest.json")
  docker exec "$candidate" python /bench/smoke_pi05_rtc_ws.py --url ws://192.168.250.1:8000 --cases /cases/cases.json --expected-weights-sha256 "$weights_sha" --expected-prefix-norm quantile --output /tmp/50h240000-smoke.json
  docker cp "$candidate:/tmp/50h240000-smoke.json" "$stage/50h240000-smoke.json"
  docker stop "$candidate"
  trap - EXIT
  exit 0
fi
[[ $(id -u) == 0 ]]
test -f "$cp_root/240000-pytorch-fp32-r1/rtc_manifest.json"
test -f "$cases/cases.json"
[[ $(docker inspect pld-rl --format '{{.State.Running}}') == false ]]
docker inspect pi05-rtc-infer --format '{{json .Config.Cmd}}' | grep -F 'rtc-50h-200000-trt-tf32-cache80-7step-20261010-r1'
! docker inspect "$candidate" >/dev/null 2>&1
! docker inspect "$preserved" >/dev/null 2>&1
docker stop pi05-rtc-infer
normal_power
if ! /usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash "$stage/rtc50h240000-deploy-20261014.sh" --gpu; then
  echo '240000 failed a GPU gate; restoring 200000 without relaxing validation' >&2
  restore_200000
  exit 1
fi
normal_power
docker rename pi05-rtc-infer "$preserved"
docker rename "$candidate" pi05-rtc-infer
if ! (
  set -e
  start_online thor-pi-maxn-50h240000-20261014 || exit 1
  docker inspect pi05-rtc-infer --format '{{json .Config.Cmd}}' | grep -F "$engine" || exit 1
  weights_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_weights_sha256"])' "$cp_root/240000-pytorch-fp32-r1/rtc_manifest.json") || exit 1
  docker exec pi05-rtc-infer python /bench/smoke_pi05_rtc_ws.py --url ws://192.168.250.1:8000 --cases /cases/cases.json --expected-weights-sha256 "$weights_sha" --expected-prefix-norm quantile --output /tmp/50h240000-online-smoke.json || exit 1
  docker cp pi05-rtc-infer:/tmp/50h240000-online-smoke.json "$stage/50h240000-online-smoke.json" || exit 1
); then
  docker stop pi05-rtc-infer
  normal_power
  docker rename pi05-rtc-infer pi05-rtc-50h240000-failed-20261014
  docker rename "$preserved" pi05-rtc-infer
  restore_200000
  echo '240000 online validation failed; restored 200000' >&2
  exit 1
fi
date --iso-8601=seconds
nvpmodel -q
df -B1 "$root"
echo DEPLOYED_50H_240000_PRESERVED_200000_AND_230000
