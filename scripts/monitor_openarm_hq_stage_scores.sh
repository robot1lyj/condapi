#!/usr/bin/env bash

set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
AUTO_RESTART="${AUTO_RESTART:-true}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
STALL_SECONDS="${STALL_SECONDS:-1200}"
ONCE=false
if [[ "${1:-}" == "--once" ]]; then
    ONCE=true
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--once]" >&2
    exit 2
fi

REPO_ROOT="/share/home/linyongjia/conda-pi/openpi"
PYTHON="/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python"
DATASETS_ROOT="/share/home/linyongjia/datasets"
OUTPUT_ROOT="/share/home/linyongjia/output/openpi/logs/openarm_kai0_stage_scores_hq_v1"
LATEST_STATUS="$OUTPUT_ROOT/monitor_latest.txt"
HISTORY_LOG="$OUTPUT_ROOT/monitor.log"
ALERT_LOG="$OUTPUT_ROOT/monitor_alerts.log"
STATE_DIR="$OUTPUT_ROOT/monitor_state"
FINAL_MARKER="$OUTPUT_ROOT/hq999_complete"
FINALIZE_LOG="$OUTPUT_ROOT/finalize_report.log"
mkdir -p "$OUTPUT_ROOT" "$STATE_DIR"

# host session shard expected gpu_id source_episode_range
JOBS=(
    "gpu12 kai0_hq_s0 s0_000_167 167 0 0:167"
    "gpu12 kai0_hq_s1 s1_167_334 167 1 167:334"
    "gpu14 kai0_hq_s2 s2_334_501 167 0 334:501"
    "gpu14 kai0_hq_s3 s3_501_668 167 1 501:668"
    "gpu28 kai0_hq_s4 s4_668_835 167 0 668:835"
    "gpu28 kai0_hq_s5 s5_835_999 164 1 835:999"
)

restart_job() {
    local host="$1" session="$2" shard="$3" gpu_id="$4" episodes="$5"
    local log_path="$OUTPUT_ROOT/${shard%%_*}_${shard#*_}.log"
    # Existing logs use exactly <shard>.log; keep that stable across restarts.
    log_path="$OUTPUT_ROOT/${shard}.log"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- \
        "$session" "$gpu_id" "$episodes" "$shard" "$REPO_ROOT" "$log_path" <<'REMOTE'
session="$1"
gpu_id="$2"
episodes="$3"
shard="$4"
repo_root="$5"
log_path="$6"

tmux kill-session -t "$session" 2>/dev/null || true
command=(
    bash scripts/run_openarm_stage_score_shard.sh
    --gpu-id "$gpu_id"
    --episodes "$episodes"
    --shard-name "$shard"
    --batch-size 32
    --resume
)
printf -v quoted_command '%q ' "${command[@]}"
tmux new-session -d -s "$session" \
    "cd $(printf '%q' "$repo_root") && exec $quoted_command >>$(printf '%q' "$log_path") 2>&1"
REMOTE
}

finalize_report() {
    [[ -e "$FINAL_MARKER" ]] && return 0
    ssh -o BatchMode=yes -o ConnectTimeout=10 gpu28 bash -s -- \
        "$REPO_ROOT" "$PYTHON" "$DATASETS_ROOT" >"$FINALIZE_LOG" 2>&1 <<'REMOTE'
repo_root="$1"
python="$2"
datasets_root="$3"
cd "$repo_root"
"$python" scripts/build_openarm_hq_score_report.py \
    --datasets-root "$datasets_root" \
    --output-root "$datasets_root/openarm_hq_score_review_v1" \
    --overwrite
REMOTE
    touch "$FINAL_MARKER"
}

while true; do
    temporary_status="${LATEST_STATUS}.tmp.$$"
    timestamp="$(date '+%F %T %Z')"
    total_completed=0
    all_complete=true
    {
        echo "timestamp=$timestamp"
        for job in "${JOBS[@]}"; do
            read -r host session shard expected gpu_id episodes <<<"$job"
            log_path="$OUTPUT_ROOT/${shard}.log"
            result="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" bash -s -- \
                "$session" "$shard" "$expected" "$STALL_SECONDS" "$log_path" <<'REMOTE'
session="$1"
shard="$2"
expected="$3"
stall_seconds="$4"
log_path="$5"
count="$(find "/share/home/linyongjia/datasets/openarm_kai0_stage_scores_hq_v1_${shard}/data" -name '*.parquet' 2>/dev/null | wc -l)"
if [[ "$count" -ge "$expected" ]]; then
    state="complete"
elif tmux has-session -t "$session" 2>/dev/null; then
    now="$(date +%s)"
    log_mtime="$(stat -c %Y "$log_path" 2>/dev/null || echo 0)"
    if (( log_mtime > 0 && now - log_mtime > stall_seconds )); then
        state="stalled"
    else
        state="running"
    fi
else
    state="stopped"
fi
printf '%s %s\n' "$state" "$count"
REMOTE
)" || result="unreachable 0"
            read -r state count <<<"$result"
            total_completed=$((total_completed + count))

            state_file="$STATE_DIR/$session"
            previous_count=0
            restart_count=0
            if [[ -s "$state_file" ]]; then
                read -r previous_count restart_count <"$state_file"
            fi
            if (( count > previous_count )); then
                restart_count=0
            fi

            if [[ "$state" != "complete" ]]; then
                all_complete=false
            fi
            if [[ "$state" == "stopped" || "$state" == "stalled" ]]; then
                if [[ "$AUTO_RESTART" == "true" && "$restart_count" -lt "$MAX_RESTARTS" ]]; then
                    if restart_job "$host" "$session" "$shard" "$gpu_id" "$episodes"; then
                        restart_count=$((restart_count + 1))
                        state="restarted"
                    else
                        state="restart_failed"
                    fi
                fi
                printf '%s session=%s host=%s state=%s completed=%s/%s restarts=%s\n' \
                    "$timestamp" "$session" "$host" "$state" "$count" "$expected" "$restart_count" >>"$ALERT_LOG"
            elif [[ "$state" == "unreachable" ]]; then
                printf '%s session=%s host=%s state=unreachable completed=%s/%s\n' \
                    "$timestamp" "$session" "$host" "$count" "$expected" >>"$ALERT_LOG"
            fi
            printf '%s %s\n' "$count" "$restart_count" >"$state_file"
            printf '%s host=%s state=%s completed=%s/%s restarts=%s\n' \
                "$session" "$host" "$state" "$count" "$expected" "$restart_count"
        done
        printf 'total=%s/999\n' "$total_completed"
    } >"$temporary_status"
    mv "$temporary_status" "$LATEST_STATUS"
    cat "$LATEST_STATUS" >>"$HISTORY_LOG"
    echo >>"$HISTORY_LOG"

    if $all_complete; then
        if finalize_report; then
            echo "final_report=complete" >>"$LATEST_STATUS"
        else
            echo "final_report=failed" >>"$LATEST_STATUS"
            printf '%s final_report=failed\n' "$timestamp" >>"$ALERT_LOG"
            all_complete=false
        fi
    fi

    if $ONCE || $all_complete; then
        cat "$LATEST_STATUS"
        break
    fi
    sleep "$INTERVAL_SECONDS"
done
