#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-openpi_dev_sudo}"
NAME="${NAME:-openpi_dev}"
DATA_DIR="${OPENPI_DATA_HOME:-/share/home/linyongjia/.cache/openpi/openpi-assets}"
LEROBOT_DATA_DIR="${LEROBOT_DATA_DIR:-/share/home/linyongjia/data}"
OUTPUT_DIR="${OPENPI_OUTPUT_DIR:-/share/home/linyongjia/output/openpi}"
OPENPI_DATA_HOME_IN_CONTAINER="${OPENPI_DATA_HOME_IN_CONTAINER:-/root/.cache/openpi}"
OPENPI_ASSETS_MOUNT="${OPENPI_ASSETS_MOUNT:-${OPENPI_DATA_HOME_IN_CONTAINER}/openpi-assets}"
USER_NAME="${USER_NAME:-$(id -un)}"
HOME_DIR="${HOME_DIR:-/home/linyongjia}"
WORKDIR="${WORKDIR:-/app}"
GPU_SPEC="${GPU_SPEC:-all}"
DETACH=0
HOST_NET=0
DEFAULT_PORT=6666
PORTS=()
MOUNT_SRC="${MOUNT_SRC:-$REPO_ROOT}"
EXTRA_MOUNTS=()
USER_SPEC="${USER_SPEC:-$(id -u):$(id -g)}"
RUN_AS_ROOT=0
COMMAND=()
USER_UID=""
USER_GID=""
USER_GROUP_NAME=""
OPENPI_OUTPUT_DIR_IN_CONTAINER="/output/openpi"
ENTER_EXISTING=0

usage() {
  cat <<'EOF'
Usage: scripts/docker/run_dev.sh [options] [-- <command>]

Options:
  --image <name>        Docker image to run (default: openpi_dev_sudo)
  --name <name>         Container name (default: openpi_dev)
  -p, --port <port>     Expose port (repeatable). "8000" or "8000:8000".
  --gpus <spec>         GPU spec: "all", "count=N", "N", or "0,1".
  --no-gpu              Disable GPU access.
  --data <path>         Host path for openpi-assets cache (default: /share/home/linyongjia/.cache/openpi/openpi-assets)
  --lerobot-data <path> Host path for LeRobot datasets (default: /share/home/linyongjia/data)
  --output <path>       Host path for training outputs (default: /share/home/linyongjia/output/openpi)
  --mount <path>        Bind-mount repo/workspace to /app (default: repo root)
  --bind <a:b>          Extra bind mount (repeatable), e.g. /host/file:/container/file
  --user <uid:gid>      Run as this UID:GID (default: 1110:1011 for linyongjia)
  --as-root             Run as root (disables --user)
  --enter               Enter an existing container (start if needed).
  --workdir <path>      Container working directory (default: /app)
  --host-net            Use host network (ignores -p/--port).
  -d, --detach          Run container in background.
  -h, --help            Show help.

Examples:
  scripts/docker/run_dev.sh -p 8000 --gpus all
  scripts/docker/run_dev.sh --gpus 0,1 --data /data/openpi -- /bin/bash
  scripts/docker/run_dev.sh --enter -- /bin/bash
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    -p|--port)
      PORTS+=("$2")
      shift 2
      ;;
    --gpus|--gpu)
      GPU_SPEC="$2"
      shift 2
      ;;
    --no-gpu)
      GPU_SPEC=""
      shift
      ;;
    --data)
      DATA_DIR="$2"
      shift 2
      ;;
    --lerobot-data)
      LEROBOT_DATA_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --mount)
      MOUNT_SRC="$2"
      shift 2
      ;;
    --bind)
      EXTRA_MOUNTS+=("$2")
      shift 2
      ;;
    --user)
      USER_SPEC="$2"
      RUN_AS_ROOT=0
      shift 2
      ;;
    --as-root)
      RUN_AS_ROOT=1
      USER_SPEC=""
      shift
      ;;
    --enter)
      ENTER_EXISTING=1
      shift
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --host-net)
      HOST_NET=1
      shift
      ;;
    -d|--detach)
      DETACH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      COMMAND=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if (( RUN_AS_ROOT == 0 )) && [[ "$OPENPI_DATA_HOME_IN_CONTAINER" == "/root/.cache/openpi" ]]; then
  OPENPI_DATA_HOME_IN_CONTAINER="/openpi_cache"
  OPENPI_ASSETS_MOUNT="${OPENPI_DATA_HOME_IN_CONTAINER}/openpi-assets"
fi

if (( RUN_AS_ROOT == 0 )); then
  if [[ "$USER_SPEC" != *:* ]]; then
    echo "Invalid --user value (expected uid:gid): $USER_SPEC" >&2
    exit 1
  fi
  USER_UID="${USER_SPEC%%:*}"
  USER_GID="${USER_SPEC##*:}"
  if [[ -z "$USER_UID" || -z "$USER_GID" ]]; then
    echo "Invalid --user value (expected uid:gid): $USER_SPEC" >&2
    exit 1
  fi
  USER_GROUP_NAME="${USER_GROUP_NAME:-$(getent group "$USER_GID" | awk -F: '{print $1}')}"
  if [[ -z "$USER_GROUP_NAME" ]]; then
    USER_GROUP_NAME="$USER_NAME"
  fi
fi

if (( ENTER_EXISTING )); then
  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
    echo "Container \"$NAME\" does not exist. Create it first." >&2
    exit 1
  fi
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$NAME"; then
    docker start "$NAME" >/dev/null
  fi
  if [[ ${#COMMAND[@]} -eq 0 ]]; then
    COMMAND=(/bin/bash)
  fi
  if (( RUN_AS_ROOT == 0 )); then
    command_str=$(printf '%q ' "${COMMAND[@]}")
    command_str=${command_str% }
    bootstrap_cmd=$(cat <<EOF
set -euo pipefail
if ! getent group "$USER_GID" >/dev/null; then
  echo "${USER_GROUP_NAME}:x:${USER_GID}:" >> /etc/group
fi
if ! getent passwd "$USER_UID" >/dev/null; then
  echo "${USER_NAME}:x:${USER_UID}:${USER_GID}:,,,:${HOME_DIR}:/bin/bash" >> /etc/passwd
fi
echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
chmod 440 /etc/sudoers.d/${USER_NAME}
mkdir -p \
  "${HOME_DIR}" \
  "${HOME_DIR}/.cache/jax" \
  "${OPENPI_DATA_HOME_IN_CONTAINER}" \
  "${OPENPI_OUTPUT_DIR_IN_CONTAINER}"
chown "${USER_UID}:${USER_GID}" \
  "${HOME_DIR}" \
  "${HOME_DIR}/.cache" \
  "${HOME_DIR}/.cache/jax" \
  "${OPENPI_DATA_HOME_IN_CONTAINER}" \
  "${OPENPI_OUTPUT_DIR_IN_CONTAINER}"
export HOME="${HOME_DIR}" USER="${USER_NAME}" LOGNAME="${USER_NAME}"
exec su -m -s /bin/bash "${USER_NAME}" -c "cd \"${WORKDIR}\" && exec ${command_str}"
EOF
)
    exec docker exec -it -u 0 "$NAME" /bin/bash -lc "$bootstrap_cmd"
  fi
  exec docker exec -it -u 0 -w "$WORKDIR" "$NAME" "${COMMAND[@]}"
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "Container \"$NAME\" already exists. Remove it or choose a new name." >&2
  exit 1
fi

mkdir -p "$DATA_DIR"
mkdir -p "$LEROBOT_DATA_DIR"
mkdir -p "$OUTPUT_DIR"

run_args=(
  --name "$NAME"
  --init
  -e OPENPI_DATA_HOME="$OPENPI_DATA_HOME_IN_CONTAINER"
  -e OPENPI_OUTPUT_DIR=/output/openpi
  -e HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data}"
  -e IS_DOCKER=true
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -v "$DATA_DIR":"$OPENPI_ASSETS_MOUNT"
  -v "$LEROBOT_DATA_DIR":/data
  -v "$OUTPUT_DIR":/output/openpi
  -w "$WORKDIR"
)

if (( RUN_AS_ROOT == 0 )); then
  run_args+=(-e HOME="$HOME_DIR" -e USER="$USER_NAME" -e LOGNAME="$USER_NAME")
fi

if [[ -n "$MOUNT_SRC" ]]; then
  run_args+=(-v "$MOUNT_SRC":/app)
fi
for mount in "${EXTRA_MOUNTS[@]}"; do
  run_args+=(-v "$mount")
done

if (( HOST_NET )); then
  run_args+=(--network host)
fi

if (( DETACH )); then
  run_args+=(-d)
else
  run_args+=(-it)
fi

if (( HOST_NET == 0 )); then
  if [[ ${#PORTS[@]} -eq 0 ]]; then
    PORTS+=("$DEFAULT_PORT")
  fi
  for p in "${PORTS[@]}"; do
    if [[ "$p" == *:* ]]; then
      run_args+=(-p "$p")
    else
      run_args+=(-p "$p:$p")
    fi
  done
fi

if [[ -n "$GPU_SPEC" ]]; then
  if [[ "$GPU_SPEC" == "all" ]]; then
    run_args+=(--gpus all)
  elif [[ "$GPU_SPEC" =~ ^count=[0-9]+$ ]]; then
    run_args+=(--gpus "$GPU_SPEC")
  elif [[ "$GPU_SPEC" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    run_args+=(--gpus "device=$GPU_SPEC")
  elif [[ "$GPU_SPEC" =~ ^[0-9]+$ ]]; then
    run_args+=(--gpus "count=$GPU_SPEC")
  else
    run_args+=(--gpus "$GPU_SPEC")
  fi
fi

if (( DETACH )) && [[ ${#COMMAND[@]} -eq 0 ]]; then
  COMMAND=(sleep infinity)
fi
if [[ ${#COMMAND[@]} -eq 0 ]]; then
  COMMAND=(/bin/bash)
fi

RUN_COMMAND=("${COMMAND[@]}")
if (( RUN_AS_ROOT == 0 )); then
  command_str=$(printf '%q ' "${COMMAND[@]}")
  command_str=${command_str% }
  bootstrap_cmd=$(cat <<EOF
set -euo pipefail
if ! getent group "$USER_GID" >/dev/null; then
  echo "${USER_GROUP_NAME}:x:${USER_GID}:" >> /etc/group
fi
if ! getent passwd "$USER_UID" >/dev/null; then
  echo "${USER_NAME}:x:${USER_UID}:${USER_GID}:,,,:${HOME_DIR}:/bin/bash" >> /etc/passwd
fi
echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
chmod 440 /etc/sudoers.d/${USER_NAME}
mkdir -p \
  "${HOME_DIR}" \
  "${HOME_DIR}/.cache/jax" \
  "${OPENPI_DATA_HOME_IN_CONTAINER}" \
  "${OPENPI_OUTPUT_DIR_IN_CONTAINER}"
chown "${USER_UID}:${USER_GID}" \
  "${HOME_DIR}" \
  "${HOME_DIR}/.cache" \
  "${HOME_DIR}/.cache/jax" \
  "${OPENPI_DATA_HOME_IN_CONTAINER}" \
  "${OPENPI_OUTPUT_DIR_IN_CONTAINER}"
export HOME="${HOME_DIR}" USER="${USER_NAME}" LOGNAME="${USER_NAME}"
exec su -m -s /bin/bash "${USER_NAME}" -c "cd \"${WORKDIR}\" && exec ${command_str}"
EOF
)
  RUN_COMMAND=(/bin/bash -lc "$bootstrap_cmd")
fi

docker run "${run_args[@]}" "$IMAGE" "${RUN_COMMAND[@]}"
