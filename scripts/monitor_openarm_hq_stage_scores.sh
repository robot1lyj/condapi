#!/usr/bin/env bash

set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
ONCE=false
if [[ "${1:-}" == "--once" ]]; then
    ONCE=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--once]" >&2
    exit 2
fi

OUTPUT_ROOT="/share/home/linyongjia/output/openpi/logs/openarm_kai0_stage_scores_hq_v1"
LATEST_STATUS="$OUTPUT_ROOT/monitor_latest.txt"
HISTORY_LOG="$OUTPUT_ROOT/monitor.log"
ALERT_LOG="$OUTPUT_ROOT/monitor_alerts.log"
mkdir -p "$OUTPUT_ROOT"

JOBS=(
    "gpu12 kai0_hq_s0 s0_000_167 167"
    "gpu12 kai0_hq_s1 s1_167_334 167"
    "gpu14 kai0_hq_s2 s2_334_501 167"
    "gpu14 kai0_hq_s3 s3_501_668 167"
    "gpu28 kai0_hq_s4 s4_668_835 167"
    "gpu28 kai0_hq_s5 s5_835_999 164"
)

while true; do
    temporary_status="${LATEST_STATUS}.tmp.$$"
    timestamp="$(date '+%F %T %Z')"
    total_completed=0
    all_complete=true
    {
        echo "timestamp=$timestamp"
        for job in "${JOBS[@]}"; do
            read -r host session shard expected <<<"$job"
            result="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- "$session" "$shard" "$expected" <<'REMOTE'
session="$1"
shard="$2"
expected="$3"
count="$(find "/share/home/linyongjia/datasets/openarm_kai0_stage_scores_hq_v1_${shard}/data" -name '*.parquet' 2>/dev/null | wc -l)"
if [[ "$count" -ge "$expected" ]]; then
    state="complete"
elif tmux has-session -t "$session" 2>/dev/null; then
    state="running"
else
    state="stopped"
fi
printf '%s %s\n' "$state" "$count"
REMOTE
)" || result="unreachable 0"
            read -r state count <<<"$result"
            total_completed=$((total_completed + count))
            if [[ "$state" != "complete" ]]; then
                all_complete=false
            fi
            printf '%s host=%s state=%s completed=%s/%s\n' "$session" "$host" "$state" "$count" "$expected"
            if [[ "$state" == "stopped" || "$state" == "unreachable" ]]; then
                printf '%s session=%s host=%s state=%s completed=%s/%s\n' \
                    "$timestamp" "$session" "$host" "$state" "$count" "$expected" >>"$ALERT_LOG"
            fi
        done
        printf 'total=%s/999\n' "$total_completed"
    } >"$temporary_status"
    mv "$temporary_status" "$LATEST_STATUS"
    cat "$LATEST_STATUS" >>"$HISTORY_LOG"
    echo >>"$HISTORY_LOG"

    if $ONCE || $all_complete; then
        cat "$LATEST_STATUS"
        break
    fi
    sleep "$INTERVAL_SECONDS"
done
