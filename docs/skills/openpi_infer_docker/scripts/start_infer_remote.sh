#!/usr/bin/env bash
set -euo pipefail

# Start serve_policy on remote openpi_dev container using latest valid training artifact.
# Workflow:
# 1) inspect latest training artifacts
# 2) derive config/exp/step/policy_dir (+ repo_id and prompt mode for report)
# 3) start serve_policy inside container
# 4) verify service health

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROFILE="/home/lyj/.codex/skills/openpi_docker/references/server_profile.env"
PROFILE_FILE="${PROFILE_FILE:-$DEFAULT_PROFILE}"

load_profile() {
  local profile="$1"
  if [ -f "$profile" ]; then
    while IFS= read -r line; do
      line="${line%%#*}"
      line="${line%%$'\r'}"
      if [ -z "${line//[[:space:]]/}" ]; then
        continue
      fi
      if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
        local key="${line%%=*}"
        local value="${line#*=}"
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        if [ -z "${!key:-}" ]; then
          export "$key=$value"
        fi
      fi
    done < "$profile"
  fi
}

load_profile "$PROFILE_FILE"

: "${REMOTE_USER:?REMOTE_USER is required}"
: "${REMOTE_HOST:?REMOTE_HOST is required}"
: "${REMOTE_PORT:?REMOTE_PORT is required}"
: "${OPENPI_OUTPUT_DIR:?OPENPI_OUTPUT_DIR is required}"
: "${WORKDIR:=/app}"
: "${OPENPI_OUTPUT_DIR_IN_CONTAINER:=/output/openpi}"

PORT="${PORT:-6666}"
RTC_MODE_RAW="${RTC_MODE:-off}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
CONTAINER_NAME="${CONTAINER_NAME:-openpi_dev}"
FORCE_RESTART="${FORCE_RESTART:-1}"

# Optional manual overrides. If not set, use latest inspected result.
CONFIG_NAME="${CONFIG_NAME:-}"
EXP_NAME="${EXP_NAME:-}"
STEP="${STEP:-}"

inspect_json="$(bash "$SCRIPT_DIR/inspect_training_remote.sh")"

mapfile -t inspected < <(
  python3 -c '
import json
import sys
data = json.loads(sys.stdin.read())
print(str(data.get("training_ok", False)).lower())
print(data.get("config_name") or "")
print(data.get("exp_name") or "")
print("" if data.get("latest_step") is None else str(data.get("latest_step")))
print(data.get("policy_dir") or "")
print(data.get("repo_id") or "")
print(data.get("prompt_mode") or "")
print(data.get("reason") or "")
' <<<"$inspect_json"
)

training_ok="${inspected[0]}"
auto_config="${inspected[1]}"
auto_exp="${inspected[2]}"
auto_step="${inspected[3]}"
auto_policy_dir="${inspected[4]}"
auto_repo_id="${inspected[5]}"
prompt_mode="${inspected[6]}"
inspect_reason="${inspected[7]}"

if [ "$training_ok" != "true" ] && { [ -z "$CONFIG_NAME" ] || [ -z "$EXP_NAME" ] || [ -z "$STEP" ]; }; then
  echo "[err] training inspection failed: $inspect_reason"
  echo "[hint] set CONFIG_NAME/EXP_NAME/STEP manually if you want to force a specific checkpoint."
  exit 1
fi

if [ -z "$CONFIG_NAME" ]; then
  CONFIG_NAME="$auto_config"
fi
if [ -z "$EXP_NAME" ]; then
  EXP_NAME="$auto_exp"
fi
if [ -z "$STEP" ]; then
  STEP="$auto_step"
fi

if [ -z "$CONFIG_NAME" ] || [ -z "$EXP_NAME" ] || [ -z "$STEP" ]; then
  echo "[err] CONFIG_NAME/EXP_NAME/STEP are required after resolution"
  exit 1
fi

policy_dir_host="$OPENPI_OUTPUT_DIR/$CONFIG_NAME/$EXP_NAME/$STEP"
if [ -n "$auto_policy_dir" ] && [ "$CONFIG_NAME" = "$auto_config" ] && [ "$EXP_NAME" = "$auto_exp" ] && [ "$STEP" = "$auto_step" ]; then
  policy_dir_host="$auto_policy_dir"
fi

policy_dir_container="$OPENPI_OUTPUT_DIR_IN_CONTAINER/$CONFIG_NAME/$EXP_NAME/$STEP"
if [[ "$policy_dir_host" == "$OPENPI_OUTPUT_DIR/"* ]]; then
  suffix="${policy_dir_host#"$OPENPI_OUTPUT_DIR/"}"
  policy_dir_container="$OPENPI_OUTPUT_DIR_IN_CONTAINER/$suffix"
fi

default_prompt_arg=""
if [ -n "$DEFAULT_PROMPT" ]; then
  default_prompt_arg="--default-prompt $(printf '%q' "$DEFAULT_PROMPT")"
fi

if [ -z "$POLICY_REPO_ID" ] && [ -n "$auto_repo_id" ]; then
  POLICY_REPO_ID="$auto_repo_id"
fi
policy_repo_id_arg=""
if [ -n "$POLICY_REPO_ID" ]; then
  policy_repo_id_arg="--policy-repo-id $POLICY_REPO_ID"
fi

rtc_mode_upper="$(echo "$RTC_MODE_RAW" | tr '[:lower:]' '[:upper:]')"
case "$rtc_mode_upper" in
  OFF|AUTO|ONLY) ;;
  *)
    echo "[err] invalid RTC_MODE: $RTC_MODE_RAW (expected off|auto|only or OFF|AUTO|ONLY)"
    exit 1
    ;;
esac

serve_cmd="python scripts/serve_policy.py --port $PORT --rtc-mode $rtc_mode_upper $policy_repo_id_arg policy:checkpoint --policy.config=$CONFIG_NAME --policy.dir $policy_dir_container $default_prompt_arg"
serve_cmd_b64="$(printf '%s' "$serve_cmd" | base64 | tr -d '\n')"

echo "[info] repo_id=$auto_repo_id prompt_mode=$prompt_mode"
echo "[info] resolved config=$CONFIG_NAME exp=$EXP_NAME step=$STEP"
echo "[info] policy_dir_host=$policy_dir_host"
echo "[info] policy_dir_container=$policy_dir_container"
echo "[info] rtc_mode=$rtc_mode_upper"
echo "[info] policy_repo_id=${POLICY_REPO_ID:-<none>}"
echo "[info] serve_cmd=$serve_cmd"

ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" /bin/bash -s -- \
  "$CONTAINER_NAME" "$PORT" "$WORKDIR" "$FORCE_RESTART" "$serve_cmd_b64" <<'REMOTE_SCRIPT'
set -euo pipefail

container_name="$1"
port="$2"
workdir="$3"
force_restart="$4"
serve_cmd_b64="$5"
serve_cmd="$(printf '%s' "$serve_cmd_b64" | base64 -d)"

if ! docker ps --format '{{.Names}}' | grep -Fxq "$container_name"; then
  echo "[err] container not running: $container_name"
  exit 1
fi

if [ "$force_restart" = "1" ]; then
  docker exec "$container_name" bash -lc "\
    pids=\$(ps -ef | grep 'scripts/serve_policy.py' | grep -- '--port $port' | grep -v grep | awk '{print \$2}' || true); \
    if [ -n \"\$pids\" ]; then kill \$pids; fi"
fi

docker exec "$container_name" bash -lc "\
  cd '$workdir'; \
  nohup $serve_cmd > /tmp/openpi_infer_${port}.log 2>&1 & \
  echo \$! > /tmp/openpi_infer_${port}.pid"

proc_line=""
for _ in $(seq 1 30); do
  proc_line="$(docker exec "$container_name" bash -lc "ps -ef | grep 'scripts/serve_policy.py' | grep -- '--port $port' | grep -v grep | head -n 1" || true)"
  if [ -n "$proc_line" ]; then
    break
  fi
  sleep 1
done
if [ -z "$proc_line" ]; then
  echo "[err] failed to start serve_policy on port $port"
  docker exec "$container_name" bash -lc "tail -n 60 /tmp/openpi_infer_${port}.log || true"
  exit 1
fi

health=""
for _ in $(seq 1 240); do
  health="$(docker exec "$container_name" bash -lc "wget -qO- --timeout=5 http://127.0.0.1:$port/healthz" 2>/dev/null || true)"
  if [ "$health" = "OK" ]; then
    break
  fi
  sleep 1
done
if [ "$health" != "OK" ]; then
  echo "[err] server started but health check failed: ${health:-<empty>}"
  docker exec "$container_name" bash -lc "tail -n 60 /tmp/openpi_infer_${port}.log || true"
  exit 1
fi

echo "[ok] inference server started"
echo "[ok] process=$proc_line"
echo "[ok] log=/tmp/openpi_infer_${port}.log"
REMOTE_SCRIPT

bash "$SCRIPT_DIR/check_infer_remote.sh"
