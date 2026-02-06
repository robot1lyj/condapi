#!/usr/bin/env bash
set -euo pipefail

# Check whether serve_policy is running successfully on remote openpi_dev container.
# Always prints raw Docker outputs to use as user-facing completion evidence.

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

PORT="${PORT:-6666}"
CONTAINER_NAME="${CONTAINER_NAME:-openpi_dev}"
DOCKER_LOG_TAIL="${DOCKER_LOG_TAIL:-120}"
FILE_LOG_TAIL="${FILE_LOG_TAIL:-120}"

ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" /bin/bash -s -- \
  "$CONTAINER_NAME" "$PORT" "$DOCKER_LOG_TAIL" "$FILE_LOG_TAIL" <<'REMOTE_SCRIPT'
set -euo pipefail

container_name="$1"
port="$2"
docker_log_tail="$3"
file_log_tail="$4"

if ! docker ps --format '{{.Names}}' | grep -Fxq "$container_name"; then
  echo "[err] container not running: $container_name"
  exit 1
fi

echo "===== docker ps ====="
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'

echo "===== docker logs: $container_name (tail $docker_log_tail) ====="
docker logs --tail "$docker_log_tail" "$container_name" 2>&1 || true

echo "===== container file log: /tmp/openpi_infer_${port}.log (tail $file_log_tail) ====="
docker exec "$container_name" bash -lc \
  "tail -n $file_log_tail /tmp/openpi_infer_${port}.log 2>/dev/null || echo \"[missing] /tmp/openpi_infer_${port}.log\""

health="$(docker exec "$container_name" bash -lc "wget -qO- --timeout=5 http://127.0.0.1:$port/healthz" 2>/dev/null || true)"
proc_line="$(docker exec "$container_name" bash -lc \
  "ps -ef | grep -F 'scripts/serve_policy.py' | grep -- '--port $port' | grep -v grep | head -n 1" || true)"

echo "===== healthz ====="
if [ -n "$health" ]; then
  echo "$health"
else
  echo "[healthz failed]"
fi

echo
echo "===== process grep ====="
if [ -n "$proc_line" ]; then
  echo "$proc_line"
else
  echo "[process missing]"
fi

if [ -z "$proc_line" ]; then
  echo "[err] serve_policy process not found on port $port"
  exit 1
fi
if [ "$health" != "OK" ]; then
  echo "[err] health check failed on port $port, got: ${health:-<empty>}"
  exit 1
fi

echo "[ok] inference server is healthy"
echo "[ok] container=$container_name port=$port"
REMOTE_SCRIPT
