#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/artifacts/pi-conda-offline-bundle}"
CONDA_SPECS_FILE="$ROOT_DIR/scripts/conda/conda-specs-linux-64.txt"
PIP_REQ_FILE="$ROOT_DIR/scripts/conda/requirements-pi-pip.txt"
META_DIR="$OUT_DIR/meta"
CONDA_PKGS_DIR="$OUT_DIR/conda_pkgs"
WHEELHOUSE_DIR="$OUT_DIR/wheelhouse"
SRC_DIR="$OUT_DIR/src"
CACHE_DIR="$OUT_DIR/.cache"
PIP_CACHE_DIR="$CACHE_DIR/pip"
XDG_CACHE_DIR="$CACHE_DIR/xdg"
CONDA_CACHE_DIR="$CACHE_DIR/conda_pkgs"
TMP_ENV_DIR=""
TMP_CLONE_DIR=""
BUILD_PYTHON=""

LEROBOT_GIT_URL="https://github.com/huggingface/lerobot"
LEROBOT_GIT_REV="0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
DLIMP_GIT_URL="https://github.com/kvablack/dlimp"
DLIMP_GIT_REV="ad72ce3a9b414db2185bc0b38461d4101a65477a"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OUT_DIR]

Build an offline install bundle for the non-container conda training flow.

Output layout:
  conda_pkgs/   Exact conda package archives for linux-64.
  wheelhouse/   Python wheels for offline pip install.
  src/          Source checkouts for pinned git dependencies.
  meta/         Explicit manifests used by the installer.

Example:
  bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[err] missing required command: $1" >&2
    exit 1
  fi
}

find_conda_sh() {
  if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    local conda_base
    conda_base="$("$CONDA_EXE" info --base)"
    echo "$conda_base/etc/profile.d/conda.sh"
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    echo "$conda_base/etc/profile.d/conda.sh"
    return 0
  fi

  local candidate
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"
  do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

cleanup() {
  if [ -n "${TMP_ENV_DIR:-}" ] && [ -d "$TMP_ENV_DIR" ]; then
    rm -rf "$TMP_ENV_DIR"
  fi
  if [ -n "${TMP_CLONE_DIR:-}" ] && [ -d "$TMP_CLONE_DIR" ]; then
    rm -rf "$TMP_CLONE_DIR"
  fi
}
trap cleanup EXIT

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

require_cmd git
require_cmd python3
CONDA_SH="$(find_conda_sh)"
if [ ! -f "$CONDA_SH" ]; then
  echo "[err] could not locate conda.sh" >&2
  exit 1
fi
source "$CONDA_SH"
require_cmd conda

mkdir -p "$META_DIR" "$CONDA_PKGS_DIR" "$WHEELHOUSE_DIR" "$SRC_DIR"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_DIR" "$CONDA_CACHE_DIR"
TMP_ENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pi-conda-env.XXXXXX")"
TMP_CLONE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pi-conda-src.XXXXXX")"
export CONDA_PKGS_DIRS="$CONDA_CACHE_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_DIR"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL="${OPENPI_PIP_INDEX_URL:-https://pypi.org/simple}"
CONDA_SOLVER="${OPENPI_CONDA_SOLVER:-libmamba}"

echo "[info] output dir: $OUT_DIR"
echo "[info] solving conda environment into temp prefix: $TMP_ENV_DIR"
echo "[info] conda solver: $CONDA_SOLVER"
echo "[info] pip index: $PIP_INDEX_URL"
conda create -y --solver "$CONDA_SOLVER" -p "$TMP_ENV_DIR" -c conda-forge --override-channels --file "$CONDA_SPECS_FILE"
BUILD_PYTHON="$TMP_ENV_DIR/bin/python"

echo "[info] exporting explicit conda package list"
conda list --explicit -p "$TMP_ENV_DIR" > "$META_DIR/conda-explicit-urls.txt"
cp "$CONDA_SPECS_FILE" "$META_DIR/conda-specs-linux-64.txt"

echo "[info] collecting conda archives"
python3 - "$META_DIR/conda-explicit-urls.txt" "$CONDA_PKGS_DIR" <<'PY'
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

explicit_file = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
target_dir.mkdir(parents=True, exist_ok=True)

info = json.loads(subprocess.check_output(["conda", "info", "--json"], text=True))
pkgs_dirs = [Path(p) for p in info["pkgs_dirs"]]
copied = []

for raw_line in explicit_file.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or line.startswith("@"):
        continue
    parsed = urlparse(line)
    basename = Path(parsed.path).name
    for pkgs_dir in pkgs_dirs:
        candidate = pkgs_dir / basename
        if candidate.exists():
            shutil.copy2(candidate, target_dir / basename)
            copied.append(basename)
            break
    else:
        raise FileNotFoundError(f"could not find conda archive in cache: {basename}")

(target_dir.parent / "meta" / "conda-package-basenames.txt").write_text("\n".join(copied) + "\n")
PY

echo "[info] cloning pinned source dependencies"
git clone "$LEROBOT_GIT_URL" "$TMP_CLONE_DIR/lerobot"
git -C "$TMP_CLONE_DIR/lerobot" checkout "$LEROBOT_GIT_REV"
rm -rf "$TMP_CLONE_DIR/lerobot/.git"
cp -a "$TMP_CLONE_DIR/lerobot" "$SRC_DIR/lerobot"

git clone "$DLIMP_GIT_URL" "$TMP_CLONE_DIR/dlimp"
git -C "$TMP_CLONE_DIR/dlimp" checkout "$DLIMP_GIT_REV"
rm -rf "$TMP_CLONE_DIR/dlimp/.git"
cp -a "$TMP_CLONE_DIR/dlimp" "$SRC_DIR/dlimp"

cat > "$META_DIR/source-commits.txt" <<EOF
lerobot $LEROBOT_GIT_REV
dlimp $DLIMP_GIT_REV
EOF

echo "[info] building wheelhouse from pinned requirements"
"$BUILD_PYTHON" -m pip wheel \
  --wheel-dir "$WHEELHOUSE_DIR" \
  --retries 8 \
  --resume-retries 8 \
  --timeout 60 \
  -r "$PIP_REQ_FILE"

echo "[info] installing local build backends into temp env"
"$BUILD_PYTHON" -m pip install \
  --no-index \
  --find-links "$WHEELHOUSE_DIR" \
  hatchling \
  poetry-core

echo "[info] building wheels for source dependencies"
"$BUILD_PYTHON" -m pip wheel \
  --wheel-dir "$WHEELHOUSE_DIR" \
  --no-build-isolation \
  --no-deps \
  --retries 8 \
  --resume-retries 8 \
  --timeout 60 \
  "$SRC_DIR/lerobot" \
  "$SRC_DIR/dlimp"

cp "$PIP_REQ_FILE" "$META_DIR/requirements-pi-pip.txt"

cat > "$META_DIR/README.txt" <<EOF
Offline bundle for openpi conda training.

Install on the training server with:
  bash scripts/conda/install_offline_bundle.sh \\
    --bundle-dir <bundle_dir> \\
    --openpi-dir <server_openpi_repo> \\
    --env-name pi-conda
EOF

echo "[ok] bundle created:"
echo "  conda packages : $CONDA_PKGS_DIR"
echo "  wheelhouse     : $WHEELHOUSE_DIR"
echo "  source deps    : $SRC_DIR"
echo "  metadata       : $META_DIR"
