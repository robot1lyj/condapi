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
: "${ROOT:?ROOT is required}"
: "${OPENPI_OUTPUT_DIR:?OPENPI_OUTPUT_DIR is required}"

PORT="${PORT:-6666}"
RTC_MODE="${RTC_MODE:-off}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
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

policy_dir="$OPENPI_OUTPUT_DIR/$CONFIG_NAME/$EXP_NAME/$STEP"
if [ -n "$auto_policy_dir" ] && [ "$CONFIG_NAME" = "$auto_config" ] && [ "$EXP_NAME" = "$auto_exp" ] && [ "$STEP" = "$auto_step" ]; then
  policy_dir="$auto_policy_dir"
fi

default_prompt_arg=""
if [ -n "$DEFAULT_PROMPT" ]; then
  default_prompt_arg="--default-prompt $(printf '%q' "$DEFAULT_PROMPT")"
fi

serve_cmd="python scripts/serve_policy.py --port $PORT --rtc-mode $RTC_MODE policy:checkpoint --policy.config=$CONFIG_NAME --policy.dir $policy_dir $default_prompt_arg"

echo "[info] repo_id=$auto_repo_id prompt_mode=$prompt_mode"
echo "[info] resolved config=$CONFIG_NAME exp=$EXP_NAME step=$STEP"
echo "[info] policy_dir=$policy_dir"
echo "[info] serve_cmd=$serve_cmd"

ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" /bin/bash -s -- \
  "$CONTAINER_NAME" "$PORT" "$ROOT" "$FORCE_RESTART" "$serve_cmd" <<'REMOTE_SCRIPT'
set -euo pipefail

container_name="$1"
port="$2"
root="$3"
force_restart="$4"
serve_cmd="$5"

if ! docker ps --format '{{.Names}}' | grep -Fxq "$container_name"; then
  echo "[err] container not running: $container_name"
  exit 1
fi

if [ "$force_restart" = "1" ]; then
  docker exec "$container_name" bash -lc "\
    pids=\$(ps -ef | grep 'scripts/serve_policy.py' | grep -- '--port $port' | grep -v grep | awk '{print \$2}'); \
    if [ -n \"\$pids\" ]; then kill \$pids; fi"
fi

docker exec "$container_name" bash -lc "\
  cd '$root'; \
  nohup $serve_cmd > /tmp/openpi_infer_${port}.log 2>&1 & \
  echo \$! > /tmp/openpi_infer_${port}.pid"

sleep 2

proc_line="$(docker exec "$container_name" bash -lc "ps -ef | grep 'scripts/serve_policy.py' | grep -- '--port $port' | grep -v grep | head -n 1" || true)"
if [ -z "$proc_line" ]; then
  echo "[err] failed to start serve_policy on port $port"
  docker exec "$container_name" bash -lc "tail -n 60 /tmp/openpi_infer_${port}.log || true"
  exit 1
fi

health="$(docker exec "$container_name" bash -lc "python -c \"import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:$port/healthz', timeout=5).read().decode().strip())\"" || true)"
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
