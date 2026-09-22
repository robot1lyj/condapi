#!/bin/bash
set -euo pipefail
root=/home/wuyan-lyj/thor/openwam
run_id=${1:-r2}
[[ "$run_id" =~ ^r[0-9]+$ ]] || { echo "Expected a new run ID such as r5"; exit 2; }
if docker inspect "openwam-benchmark-$run_id" >/dev/null 2>&1 || test -e "$root/logs/benchmark-$run_id"; then
  echo "Run ID already exists; use a new ID without overwriting evidence"
  exit 2
fi
restore_pi() {
  docker stop -t 15 "openwam-benchmark-$run_id" >/dev/null 2>&1 || true
  systemd-run --unit=thor-pi-maxn-20h124000 --collect \
    /usr/bin/python3 /home/wuyan-lyj/thor/pi/probes/rtc-20h-124000-20260922-r1/maxn_session.py -- \
    /bin/bash -c 'docker start pi05-rtc-infer && /usr/bin/python3 /home/wuyan-lyj/thor/pi/probes/rtc-step-sweep-20260917/watch_docker_container.py --container pi05-rtc-infer'
}
docker stop -t 20 pi05-rtc-infer
for ((i=0;i<30;i++)); do
  if ! systemctl is-active --quiet thor-pi-maxn-20h124000.service; then break; fi
  sleep 1
done
trap restore_pi EXIT
mkdir -p "$root/logs/benchmark-$run_id"
/usr/bin/python3 /home/wuyan-lyj/thor/pi/probes/rtc-20h-124000-20260922-r1/maxn_session.py -- \
  timeout 1800 docker run --name "openwam-benchmark-$run_id" --runtime=nvidia --network none --ipc=host \
    -e HF_HUB_OFFLINE=1 -e WAM_PROFILE=1 -e DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch -e WAM_ATTENTION_IMPL=sdpa \
    -v "$root/checkpoints/OpenWAM-Alpha-Sim-RoboTwin-Full:/checkpoint:ro" \
    -v "$root/build/benchmark.py:/bench.py:ro" -v "$root/logs/benchmark-$run_id:/reports" \
    --entrypoint python openwam:thor-20260922-r2 /bench.py
