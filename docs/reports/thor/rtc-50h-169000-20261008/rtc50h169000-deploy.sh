#!/bin/bash
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-50h-169000-20261008-r1
cp_root=$root/checkpoints/lego-pi05-rtc-base-50h-20260921
tag=rtc-50h-169000
suffix=20261008-r1
image=openpi-pi:thor-trained-rtc-candidate-20260916
jax_image=openpi-pi:thor-trained-rtc-jax-ref-20260917-r3
cases=$root/test-data/pi05-rtc-50h-169000-replay-20261008-r1
ref=/artifacts/$tag-jax-quantile-7step-$suffix
export_dir=/artifacts/$tag-onnx-fp32-cache80-7step-$suffix
engine=/artifacts/$tag-trt-tf32-cache80-7step-$suffix
ckpt=/checkpoints/169000-pytorch-fp32-r1
common=(--rm --runtime=nvidia --network host --ipc host -v "$stage:/bench:ro" -v "$cp_root:/checkpoints:ro" -v "$root/artifacts:/artifacts" -v "$cases:/cases:ro" -v "$root/cache:/cache" -e OPENPI_DATA_HOME=/cache --entrypoint python)
if [[ ${1:-} == --gpu ]]; then
  docker run "${common[@]}" -e JAX_PLATFORMS=cuda -e XLA_PYTHON_CLIENT_PREALLOCATE=false "$jax_image" /bench/rtc_jax_reference.py --checkpoint /checkpoints/169000 --training-contract /checkpoints/169000/training_contract.json --cases /cases/cases.json --output "$ref" --num-steps 7
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/export_pi05_rtc_onnx.py --checkpoint "$ckpt" --cases /cases/cases.json --jax-reference "$ref" --output "$export_dir" --compute-dtype float32 --num-steps 7 --text-bucket 80 --cache-time-modulation
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/build_trt_engine.py --source "$export_dir" --output "$engine" --tf32-experiment
  docker run "${common[@]}" -e JAX_PLATFORMS=cpu "$image" /bench/validate_pi05_rtc_trt.py --engine "$engine" --checkpoint "$ckpt" --cases /cases/cases.json --jax-reference "$ref" --output "$engine/validation.json" --allow-experimental
  name=pi05-rtc-50h169000-candidate
  docker create --name "$name" --restart no --runtime=nvidia --network host --ipc host -e OPENPI_DATA_HOME=/cache -e JAX_PLATFORMS=cpu -v "$stage:/bench:ro" -v "$root/artifacts:/artifacts:ro" -v "$cp_root/169000-pytorch-fp32-r1:/ckpt:ro" -v "$cases:/cases:ro" -v "$root/cache:/cache" --entrypoint python "$image" /bench/serve_pi05_rtc_trt.py --engine "$engine" --checkpoint /ckpt --norm /ckpt/assets/yam/norm_stats.json --warmup-sample /cases/episode-000095-early.npz --warmup-rtc /bench/rtc_warmup_d0.json --host 192.168.250.1 --port 8000 --allow-validated-tf32-7step --validation "$engine/validation.json" --jax-reference "$ref"
  trap 'docker stop "$name" >/dev/null 2>&1 || true' EXIT
  docker start "$name"
  ready=false
  for tick in {1..90}; do
    [[ $(docker inspect "$name" --format '{{.State.Running}}') == true ]] || { docker logs --tail 20 "$name"; exit 1; }
    if curl -fsS --max-time 2 http://192.168.250.1:8000/healthz >/dev/null; then ready=true; break; fi
    sleep 2
  done
  [[ "$ready" == true ]]
  weights_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_weights_sha256"])' "$cp_root/169000-pytorch-fp32-r1/rtc_manifest.json")
  docker exec "$name" python /bench/smoke_pi05_rtc_ws.py --url ws://192.168.250.1:8000 --cases /cases/cases.json --expected-weights-sha256 "$weights_sha" --expected-prefix-norm quantile --output /tmp/50h169000-smoke.json
  docker cp "$name:/tmp/50h169000-smoke.json" "$stage/50h169000-smoke.json"
  docker stop "$name"
  [[ $(docker inspect pi05-rtc-infer --format '{{.State.Running}}') == false ]]
  docker rename pi05-rtc-infer pi05-rtc-50h147000-preserved-20261008
  docker rename "$name" pi05-rtc-infer
  trap - EXIT
  exit 0
fi
for tick in {1..240}; do
  [[ -f "$cp_root/169000/training_contract.json" && -f "$stage/source.sha256" ]] && break
  sleep 30
done
test -f "$cp_root/169000/training_contract.json"
(cd "$cp_root/169000" && sha256sum -c "$stage/source.sha256")
docker run --rm --network host --ipc host -e JAX_PLATFORMS=cpu -v "$stage:/bench:ro" -v "$cp_root:/checkpoints" -v "$root/artifacts:/artifacts" --entrypoint /bin/bash "$image" -c 'python /bench/audit_checkpoint_finite.py --checkpoint /checkpoints/169000 --output /artifacts/rtc-50h-169000-audit-20261008-r1.json && python /bench/prepare_rtc_checkpoint.py --checkpoint /checkpoints/169000 --training-contract /checkpoints/169000/training_contract.json --output /checkpoints/169000-pytorch-fp32-r1'
docker run --rm --network host --ipc host -e JAX_PLATFORMS=cpu -v "$stage:/bench:ro" -v "$cp_root:/checkpoints:ro" -v "$root/test-data:/data" --entrypoint python "$image" /bench/prepare_rtc_cases.py --source-suite /data/pi05-replay-v1 --source-parquet /data/ABC-130k-two-tasks/lego_sorting/train/data/source-file-000.parquet --checkpoint /checkpoints/169000 --output /data/pi05-rtc-50h-169000-replay-20261008-r1
docker stop pi05-rtc-infer
for tick in {1..30}; do
  nvpmodel -q | grep -q 'NV Power Mode: 120W' && break
  sleep 2
done
if ! nvpmodel -q | grep -q 'NV Power Mode: 120W'; then
  echo 'Old MAXN session did not exit; do not nest sessions' >&2
  exit 1
fi
if ! /usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash "$stage/rtc50h169000-deploy.sh" --gpu; then
  echo '169000 validation failed; restoring existing 147000 container' >&2
  systemd-run --unit=thor-pi-maxn-50h147000-restored --collect /usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash -c "docker start pi05-rtc-infer && /usr/bin/python3 $root/probes/rtc-step-sweep-20260917/watch_docker_container.py --container pi05-rtc-infer"
  exit 1
fi
systemd-run --unit=thor-pi-maxn-50h169000 --collect --description='Thor50h169000 RTC inference MAXN' /usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash -c "docker start pi05-rtc-infer && /usr/bin/python3 $root/probes/rtc-step-sweep-20260917/watch_docker_container.py --container pi05-rtc-infer"
echo PROMOTED_50H_169000_STARTING
ready=false
for tick in {1..90}; do
  if curl -fsS --max-time 2 http://192.168.250.1:8000/healthz >/dev/null; then ready=true; break; fi
  sleep 2
done
[[ "$ready" == true ]]
docker inspect pi05-rtc-infer --format '{{json .Config.Cmd}}' | grep -F "$engine"
weights_sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_weights_sha256"])' "$cp_root/169000-pytorch-fp32-r1/rtc_manifest.json")
docker exec pi05-rtc-infer python /bench/smoke_pi05_rtc_ws.py --url ws://192.168.250.1:8000 --cases /cases/cases.json --expected-weights-sha256 "$weights_sha" --expected-prefix-norm quantile --output /tmp/50h169000-online-smoke.json
docker cp pi05-rtc-infer:/tmp/50h169000-online-smoke.json "$stage/50h169000-online-smoke.json"
echo ONLINE_169000_VERIFIED
df -B1 "$root"
# User authorized deleting Thor-local 147000 and 136000; preserve small evidence.
docker rm pi05-rtc-50h147000-preserved-20261008 pi05-rtc-20h136000-preserved-20260930
rm -rf -- /home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-50h-20260921/147000/params /home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-20h-20260918/136000/params
rm -f -- /home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-50h-20260921/147000-pytorch-fp32-r1/model.safetensors /home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-20h-20260918/136000-pytorch-fp32-r1/model.safetensors /home/wuyan-lyj/thor/pi/artifacts/rtc-20h-136000-onnx-fp32-cache80-7step-20260923-r1/sampler.onnx.data /home/wuyan-lyj/thor/pi/artifacts/rtc-20h-136000-trt-tf32-cache80-7step-20260923-r1/sampler.engine /home/wuyan-lyj/thor/pi/artifacts/rtc-50h-147000-onnx-fp32-cache80-7step-20260930-r1/sampler.onnx.data /home/wuyan-lyj/thor/pi/artifacts/rtc-50h-147000-trt-tf32-cache80-7step-20260930-r1/sampler.engine
df -B1 "$root"
echo DEPLOYED_50H_169000_CLEANED_147000_136000
