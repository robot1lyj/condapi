#!/usr/bin/env bash
set -euo pipefail

# Run on Thor, inside maxn_session.py, after FP32 conversion and source audit.
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-20h-124000-20260922-r1
checkpoints=$root/checkpoints/lego-pi05-rtc-base-20h-20260918
cases=$root/test-data/pi05-rtc-20h-124000-replay-20260922-r1
tag=rtc-20h-136000
suffix=20260923-r1
reference=/artifacts/$tag-jax-quantile-7step-$suffix
onnx=/artifacts/$tag-onnx-fp32-cache80-7step-$suffix
engine=/artifacts/$tag-trt-tf32-cache80-7step-$suffix
weights=/checkpoints/136000-pytorch-fp32-r1
image=openpi-pi:thor-trained-rtc-candidate-20260916
jax_image=openpi-pi:thor-trained-rtc-jax-ref-20260917-r3

test -f "$checkpoints/136000-pytorch-fp32-r1/rtc_manifest.json"
test -f "$cases/cases.json"
if docker inspect -f '{{.State.Running}}' pi05-rtc-infer 2>/dev/null | grep -qx true; then
  echo 'Stop the old inference service before exclusive GPU validation' >&2
  exit 1
fi

common=(--rm --runtime=nvidia --network host --ipc host
  -v "$stage:/bench:ro"
  -v "$checkpoints:/checkpoints:ro"
  -v "$root/artifacts:/artifacts"
  -v "$cases:/cases:ro"
  -v "$root/cache:/cache"
  -e OPENPI_DATA_HOME=/cache --entrypoint python)

docker run "${common[@]}" -e JAX_PLATFORMS=cuda -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$jax_image" /bench/rtc_jax_reference.py \
  --checkpoint /checkpoints/136000 \
  --training-contract /checkpoints/136000/training_contract.json \
  --cases /cases/cases.json --output "$reference" --num-steps 7
docker run "${common[@]}" -e JAX_PLATFORMS=cpu \
  "$image" /bench/export_pi05_rtc_onnx.py \
  --checkpoint "$weights" --cases /cases/cases.json \
  --jax-reference "$reference" --output "$onnx" \
  --compute-dtype float32 --num-steps 7 --text-bucket 80 --cache-time-modulation
docker run "${common[@]}" -e JAX_PLATFORMS=cpu \
  "$image" /bench/build_trt_engine.py \
  --source "$onnx" --output "$engine" --tf32-experiment
docker run "${common[@]}" -e JAX_PLATFORMS=cpu \
  "$image" /bench/validate_pi05_rtc_trt.py \
  --engine "$engine" --checkpoint "$weights" --cases /cases/cases.json \
  --jax-reference "$reference" --output "$engine/validation.json" --allow-experimental
echo VALIDATED_20H_136000
