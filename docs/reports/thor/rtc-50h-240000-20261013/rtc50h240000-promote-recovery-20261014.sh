#!/bin/bash
# Separate frozen recovery: original deployment evidence is never overwritten.
set -euo pipefail
root=/home/wuyan-lyj/thor/pi
stage=$root/probes/rtc-50h-240000-20261014-r1
old_unit=thor-pi-maxn-50h240000-20261014.service
final_unit=thor-pi-maxn-50h240000-final-20261014
preserved=pi05-rtc-50h200000-preserved-20261014
candidate=pi05-rtc-50h240000-failed-20261014
wait_idle() {
  for tick in {1..25}; do
    if ! systemctl is-active --quiet "$1" && nvpmodel -q | grep -q 'NV Power Mode: 120W'; then return 0; fi
    sleep 2
  done
  echo 'Previous MAXN owner has not fully exited' >&2
  return 1
}
start_and_check() {
  local unit=$1 weights=$2 output=$3 id
  id=$(docker inspect pi05-rtc-infer --format '{{.Id}}') || return 1
  # Watch immutable ID, not the reusable service name, across future handoffs.
  systemd-run --unit="$unit" /usr/bin/python3 "$stage/maxn_session.py" -- /bin/bash -c "docker start $id && exec /usr/bin/python3 $root/probes/rtc-step-sweep-20260917/watch_docker_container.py --container $id" || return 1
  local healthy=false
  for tick in {1..90}; do
    # systemd-run returns before docker start: tolerate asynchronous startup.
    if [[ $(docker inspect "$id" --format '{{.State.Running}}') == true ]] && curl -fsS --max-time 2 http://192.168.250.1:8000/healthz >/dev/null; then
      healthy=true
      break
    fi
    if systemctl is-failed --quiet "$unit.service"; then break; fi
    sleep 2
  done
  [[ $healthy == true ]] || { docker logs --tail 40 "$id"; return 1; }
  docker exec "$id" python /bench/smoke_pi05_rtc_ws.py --url ws://192.168.250.1:8000 --cases /cases/cases.json --expected-weights-sha256 "$weights" --expected-prefix-norm quantile --output "/tmp/$output" || return 1
  docker cp "$id:/tmp/$output" "$stage/$output" || return 1
}
[[ $(id -u) == 0 ]]
[[ $(docker inspect pld-rl --format '{{.State.Running}}') == false ]]
docker inspect pi05-rtc-infer --format '{{json .Config.Cmd}}' | grep -F 'rtc-50h-200000-trt-tf32-cache80-7step-20261010-r1'
docker inspect "$candidate" --format '{{json .Config.Cmd}}' | grep -F 'rtc-50h-240000-trt-tf32-cache80-7step-20261014-r1'
! docker inspect "$preserved" >/dev/null 2>&1
docker stop pi05-rtc-infer
wait_idle "$old_unit"
docker rename pi05-rtc-infer "$preserved"
docker rename "$candidate" pi05-rtc-infer
if ! start_and_check "$final_unit" 12b08533d61119bf9b7b98bc20a3f64553b42a7140fd99107f52610f19184288 50h240000-final-online-smoke.json; then
  docker stop pi05-rtc-infer
  wait_idle "$final_unit.service"
  docker rename pi05-rtc-infer pi05-rtc-50h240000-recovery-failed-20261014
  docker rename "$preserved" pi05-rtc-infer
  start_and_check thor-pi-maxn-50h200000-recovery-final-20261014 298714be810f5e7e822bcac55c5efe55789f1c96c0ac4ed0a1caa86f2670e42a 200000-recovery-final-smoke.json
  echo '240000 recovery failed; 200000 restored and checked' >&2
  exit 1
fi
date --iso-8601=seconds
nvpmodel -q
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
docker inspect pi05-rtc-infer --format '{{json .Config.Cmd}}'
df -B1 "$root"
echo DEPLOYED_50H_240000_RECOVERY_VERIFIED_PRESERVED_200000_AND_230000
