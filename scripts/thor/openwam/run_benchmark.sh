#!/bin/bash
set -euo pipefail
root=/home/wuyan-lyj/thor/openwam
run_id=${1:-r2}
bench_file=${2:-benchmark.py}
compile_mode=${3:-default}
step_sweep=${4:-0}
prefix_compile=${5:-0}
prefix_case=${6:-all}
case "$step_sweep" in 0|1) ;; *) echo "Expected step sweep 0 or 1"; exit 2;; esac
case "$prefix_compile" in 0|1) ;; *) echo "Expected prefix compile 0 or 1"; exit 2;; esac
case "$prefix_case" in all|baseline|split_only|prefix_only|prefix_split) ;; *) echo "Unknown prefix case"; exit 2;; esac
case "$compile_mode" in default|reduce-overhead|max-autotune) ;; *) echo "Unknown compile mode"; exit 2;; esac
case "$bench_file" in benchmark.py|benchmark-r9.py|quant_benchmark.py|probe_attention.py|probe_prefix_invariance.py|probe_segmented_attention.py|benchmark_exact_prefix.py) ;; *) echo "Unknown benchmark script"; exit 2;; esac
[[ "$run_id" =~ ^r[0-9]+$ ]] || { echo "Expected a new run ID such as r5"; exit 2; }
if docker inspect "openwam-benchmark-$run_id" >/dev/null 2>&1 || test -e "$root/logs/benchmark-$run_id"; then
  echo "Run ID already exists; use a new ID without overwriting evidence"
  exit 2
fi
cleanup() {
  docker stop -t 15 "openwam-benchmark-$run_id" >/dev/null 2>&1 || true
  # User priority: keep Pi stopped while OpenWAM acceleration is in progress.
  # maxn_session.py restores idle power/clocks independently on exit.
}
docker stop -t 20 pi05-rtc-infer
for ((i=0;i<30;i++)); do
  if ! systemctl is-active --quiet thor-pi-maxn-20h124000.service; then break; fi
  sleep 1
done
trap cleanup EXIT
mkdir -p "$root/logs/benchmark-$run_id"
/usr/bin/python3 /home/wuyan-lyj/thor/pi/probes/rtc-20h-124000-20260922-r1/maxn_session.py -- \
  timeout 1800 docker run --name "openwam-benchmark-$run_id" --runtime=nvidia --network none --ipc=host \
    -e HF_HUB_OFFLINE=1 -e WAM_PROFILE=1 -e DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch -e WAM_ATTENTION_IMPL=sdpa \
    -e OPENWAM_COMPILE_MODE="$compile_mode" -e TORCH_LOGS=perf_hints \
    -e OPENWAM_STEP_SWEEP="$step_sweep" \
    -e OPENWAM_PREFIX_COMPILE="$prefix_compile" -e OPENWAM_PREFIX_CASE="$prefix_case" \
    -v "$root/checkpoints/OpenWAM-Alpha-Sim-RoboTwin-Full:/checkpoint:ro" \
    -v "$root/build/$bench_file:/bench.py:ro" -v "$root/logs/benchmark-$run_id:/reports" \
    --entrypoint python openwam:thor-20260922-r2 /bench.py
