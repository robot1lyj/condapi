#!/bin/bash
# User authorized conversion, exclusive GPU validation and same Pi service.
# Freeze before running. Temporary benchmark containers are removed; no new
# DAgger image/service family. Preserve 240000 and all older model artifacts.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-separation-r1-7000-20261015-r2
cp_root=$root/checkpoints/lego-pi05-rtc-separation-r1-10k-20261014
image=openpi-pi:thor-trained-rtc-candidate-20260916
jax_image=openpi-pi:thor-trained-rtc-jax-ref-20260917-r3
cases=$root/test-data/pi05-rtc-separation-r1-7000-replay-20261015-r2
ref=/artifacts/rtc-separation-r1-7000-jax-quantile-7step-20261015-r2
export_dir=/artifacts/rtc-separation-r1-7000-onnx-fp32-cache80-7step-20261015-r2
engine=/artifacts/rtc-separation-r1-7000-trt-tf32-cache80-7step-20261015-r2
temporary=pi05-rtc-validation-20261015
preserved=pi05-rtc-50h240000-preserved-20261015
online_unit=thor-pi-maxn-separation7000-20261015
common=(--rm --runtime=nvidia --network host --ipc host -v "$stage:/bench:ro" -v "$cp_root:/checkpoints:ro" -v "$root/artifacts:/artifacts" -v "$cases:/cases:ro" -v "$root/cache:/cache" -e OPENPI_DATA_HOME=/cache --entrypoint python)
service=(--restart no --runtime=nvidia --network host --ipc host -e OPENPI_DATA_HOME=/cache -e JAX_PLATFORMS=cpu -v "$stage:/bench:ro" -v "$root/artifacts:/artifacts:ro" -v "$cp_root/7000-pytorch-fp32-r1:/ckpt:ro" -v "$cases:/cases:ro" -v "$root/cache:/cache" --entrypoint python "$image" /bench/serve_pi05_rtc_trt.py --engine "$engine" --checkpoint /ckpt --norm /ckpt/assets/yam/norm_stats.json --warmup-sample /cases/episode-000095-early.npz --warmup-rtc /bench/rtc_warmup_d0.json --host 192.168.250.1 --port 8000 --allow-validated-tf32-7step --validation "$engine/validation.json" --jax-reference "$ref")
ready() {
  local id=$1
  for tick in {1..90}; do
    if [[ $(docker inspect "$id" --format '{{.State.Running}}') == true ]] && curl -fsS --max-time 2 http://192.168.250.1:8000/healthz >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  docker logs --tail 40 "$id"
  return 1
}
wait_idle() {
  for tick in {1..25}; do
    if ! systemctl is-active --quiet "$1" && nvpmodel -q | grep -q 'NV Power Mode: 120W'; then return 0; fi
    sleep 2
  done
  return 1
}
start_online() {
  local unit=$1 id
  id=$(docker inspect pi05-rtc-infer --format '{{.Id}}') || return 1
  systemd-run --unit="$unit" /usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash -c "docker start $id && exec /usr/bin/python3 $stage/watch_docker_container.py --container $id" || return 1
  ready "$id"
}
smoke() {
  docker exec "$1" python /bench/smoke_pi05_rtc_ws.py --url ws://192.168.250.1:8000 --cases /cases/cases.json --expected-weights-sha256 "$2" --expected-prefix-norm quantile --output "/tmp/$3" || return 1
  docker cp "$1:/tmp/$3" "$stage/$3"
}
weights_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_weights_sha256"])' "$cp_root/7000-pytorch-fp32-r1/rtc_manifest.json")
if [[ ${1:-} == --gpu ]]; then
  echo PHASE_JAX_REFERENCE
  docker run "${common[@]}" -e JAX_PLATFORMS=cuda -e XLA_PYTHON_CLIENT_PREALLOCATE=false "$jax_image" /bench/rtc_jax_reference.py --checkpoint /checkpoints/7000 --training-contract /checkpoints/7000/training_contract.json --cases /cases/cases.json --output "$ref" --num-steps 7
  echo PHASE_TORCH_ONNX
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/export_pi05_rtc_onnx.py --checkpoint /checkpoints/7000-pytorch-fp32-r1 --cases /cases/cases.json --jax-reference "$ref" --output "$export_dir" --compute-dtype float32 --num-steps 7 --text-bucket 80 --cache-time-modulation
  echo PHASE_TENSORRT_BUILD
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/build_trt_engine.py --source "$export_dir" --output "$engine" --tf32-experiment
  echo PHASE_TENSORRT_VALIDATION
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/validate_pi05_rtc_trt.py --engine "$engine" --checkpoint /checkpoints/7000-pytorch-fp32-r1 --cases /cases/cases.json --jax-reference "$ref" --output "$engine/validation.json" --allow-experimental
  echo PHASE_TEMPORARY_PROTOCOL
  docker create --rm --name "$temporary" "${service[@]}"
  trap 'docker stop "$temporary" >/dev/null 2>&1 || true' EXIT
  docker start "$temporary"
  ready "$temporary"
  smoke "$temporary" "$weights_sha" separation7000-candidate-smoke.json
  docker stop "$temporary"
  trap - EXIT
  exit 0
fi
[[ $(id -u) == 0 ]]
[[ $(docker inspect pi05-rtc-infer --format '{{.State.Running}}') == false ]]
[[ $(docker inspect pld-rl --format '{{.State.Running}}') == false ]]
docker inspect pi05-rtc-infer --format '{{json .Config.Cmd}}' | grep -F 'rtc-50h-240000-trt-tf32-cache80-7step-20261014-r1'
nvpmodel -q | grep -q 'NV Power Mode: 120W'
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
! docker inspect "$preserved" >/dev/null 2>&1
! docker inspect "$temporary" >/dev/null 2>&1
# A GPU gate failure leaves the previous Pi stopped, as observed before work.
/usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash "$stage/deploy-20261015.sh" --gpu
nvpmodel -q | grep -q 'NV Power Mode: 120W'
docker rename pi05-rtc-infer "$preserved"
if ! (
  set -e
  docker create --name pi05-rtc-infer "${service[@]}" || exit 1
  start_online "$online_unit" || exit 1
  docker inspect pi05-rtc-infer --format '{{json .Config.Cmd}}' | grep -F "$engine" || exit 1
  smoke pi05-rtc-infer "$weights_sha" separation7000-online-smoke.json || exit 1
); then
  if docker inspect pi05-rtc-infer >/dev/null 2>&1; then
    docker stop pi05-rtc-infer || true
    wait_idle "$online_unit.service"
    docker rename pi05-rtc-infer pi05-rtc-separation7000-failed-20261015
  fi
  docker rename "$preserved" pi05-rtc-infer
  start_online thor-pi-maxn-50h240000-restored-separation7000-20261015
  smoke pi05-rtc-infer 12b08533d61119bf9b7b98bc20a3f64553b42a7140fd99107f52610f19184288 240000-restored-separation7000-smoke.json
  echo 'New online gate failed; original 240000 restored and verified' >&2
  exit 1
fi
date --iso-8601=seconds
nvpmodel -q
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
df -B1 "$root"
echo DEPLOYED_PI05_240000_PLUS_SEPARATION_7000_SAME_SERVICE
