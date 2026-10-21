#!/bin/bash
# Authorized switch to the earbuds policy; old DAgger model is preserved.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-earbuds-112000-20261021-r1
cp_root=$root/checkpoints/earbuds-pi05-rtc-base-50h-20261012
image=openpi-pi:thor-trained-rtc-candidate-20260916
jax_image=openpi-pi:thor-trained-rtc-jax-ref-20260917-r3
cases=$root/test-data/pi05-rtc-earbuds-112000-replay-20261021-r1
ref=/artifacts/rtc-earbuds-112000-jax-quantile-7step-20261021-r1
export_dir=/artifacts/rtc-earbuds-112000-onnx-fp32-cache80-7step-20261021-r1
engine=/artifacts/rtc-earbuds-112000-trt-tf32-cache80-7step-20261021-r1
temporary=pi05-rtc-validation-earbuds112000-20261021
preserved=pi05-rtc-separation10000-preserved-20261021
online_unit=thor-pi-maxn-earbuds112000-20261021
old_unit=thor-pi-maxn-separation10000-20261015.service
old_sha=19f8db38b948d71b1decce69cb267f784a94249c9ceba5a8b5cbb14881468535
common=(--rm --runtime=nvidia --network host --ipc host -v "$stage:/bench:ro" -v "$cp_root:/checkpoints:ro" -v "$root/artifacts:/artifacts" -v "$cases:/cases:ro" -v "$root/cache:/cache" -e OPENPI_DATA_HOME=/cache --entrypoint python)
service=(--restart no --runtime=nvidia --network host --ipc host -e OPENPI_DATA_HOME=/cache -e JAX_PLATFORMS=cpu -v "$stage:/bench:ro" -v "$root/artifacts:/artifacts:ro" -v "$cp_root/112000-pytorch-fp32-r1:/ckpt:ro" -v "$cases:/cases:ro" -v "$root/cache:/cache" --entrypoint python "$image" /bench/serve_pi05_rtc_trt.py --engine "$engine" --checkpoint /ckpt --norm /ckpt/assets/yam/norm_stats.json --warmup-sample /cases/episode-000000-early.npz --warmup-rtc /bench/rtc_warmup_d0.json --host 192.168.250.1 --port 8000 --allow-validated-tf32-7step --validation "$engine/validation.json" --jax-reference "$ref")
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
restore_old() {
  start_online "$1" || return 1
  smoke pi05-rtc-infer "$old_sha" "$2"
}
weights_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_weights_sha256"])' "$cp_root/112000-pytorch-fp32-r1/rtc_manifest.json")
if [[ ${1:-} == --gpu ]]; then
  echo PHASE_JAX_REFERENCE
  docker run "${common[@]}" -e JAX_PLATFORMS=cuda -e XLA_PYTHON_CLIENT_PREALLOCATE=false "$jax_image" /bench/rtc_jax_reference.py --checkpoint /checkpoints/112000 --training-contract /checkpoints/112000/training_contract.json --cases /cases/cases.json --output "$ref" --num-steps 7
  echo PHASE_TORCH_ONNX
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/export_pi05_rtc_onnx.py --checkpoint /checkpoints/112000-pytorch-fp32-r1 --cases /cases/cases.json --jax-reference "$ref" --output "$export_dir" --compute-dtype float32 --num-steps 7 --text-bucket 80 --cache-time-modulation
  echo PHASE_TENSORRT_BUILD
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/build_trt_engine.py --source "$export_dir" --output "$engine" --tf32-experiment
  echo PHASE_TENSORRT_VALIDATION
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/validate_pi05_rtc_trt.py --engine "$engine" --checkpoint /checkpoints/112000-pytorch-fp32-r1 --cases /cases/cases.json --jax-reference "$ref" --output "$engine/validation.json" --allow-experimental
  echo PHASE_TEMPORARY_PROTOCOL
  docker create --name "$temporary" "${service[@]}"
  trap 'docker stop "$temporary" >/dev/null 2>&1 || true' EXIT
  docker start "$temporary"
  ready "$temporary"
  smoke "$temporary" "$weights_sha" earbuds112000-candidate-smoke.json
  docker stop "$temporary"
  # Temporary runtime only: artifacts and logs are bind-mounted and retained.
  docker rm "$temporary"
  trap - EXIT
  exit 0
fi
[[ $(id -u) == 0 ]]
[[ $(docker inspect pi05-rtc-infer --format '{{.Id}}') == 908cf1c2b7d8a99fe3410dd84f8d01a4cdf1314b0036287e5db8899af99a4cd0 ]]
! docker inspect "$preserved" >/dev/null 2>&1
! docker inspect "$temporary" >/dev/null 2>&1
docker stop pi05-rtc-infer
wait_idle "$old_unit"
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
# Failed numerical/protocol gates restore the previous verified policy.
if ! /usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash "$stage/deploy.sh" --gpu; then
  restore_old thor-pi-maxn-separation10000-restored-earbuds112000-20261021 separation10000-restored-earbuds112000-smoke.json
  echo 'Earbuds GPU gate failed; previous policy restored' >&2
  exit 1
fi
nvpmodel -q | grep -q 'NV Power Mode: 120W'
docker rename pi05-rtc-infer "$preserved"
if ! (
  set -e
  docker create --name pi05-rtc-infer "${service[@]}" || exit 1
  start_online "$online_unit" || exit 1
  docker inspect pi05-rtc-infer --format '{{json .Config.Cmd}}' | grep -F "$engine" || exit 1
  smoke pi05-rtc-infer "$weights_sha" earbuds112000-online-smoke.json || exit 1
); then
  if docker inspect pi05-rtc-infer >/dev/null 2>&1; then
    docker stop pi05-rtc-infer || true
    wait_idle "$online_unit.service"
    docker rename pi05-rtc-infer pi05-rtc-earbuds112000-failed-20261021
  fi
  docker rename "$preserved" pi05-rtc-infer
  restore_old thor-pi-maxn-separation10000-restored-onlineearbuds-20261021 separation10000-restored-onlineearbuds-smoke.json
  echo 'Earbuds online gate failed; previous policy restored' >&2
  exit 1
fi
date --iso-8601=seconds
nvpmodel -q
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
df -B1 "$root"
echo DEPLOYED_PI05_EARBUDS_112000_SAME_SERVICE
