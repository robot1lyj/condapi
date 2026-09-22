#!/usr/bin/env bash
set -euo pipefail

# Continue an interrupted server install without replacing its existing prefix.
# The proxy is deliberately supplied by the caller because it is a temporary
# server-local relay, not a persistent environment setting.
work=${OPENWAM_INSTALL_WORK:-/home/wuyan/lyj/openwam-install-7c5861e}
prefix=${OPENWAM_INSTALL_PREFIX:-/home/wuyan/.conda/envs/vla-openwam}
code=${OPENWAM_INSTALL_CODE:-$work/code}
proxy=${PIP_PROXY:-http://127.0.0.1:18892}

[[ -x "$prefix/bin/python" ]] || { echo "missing Python in $prefix" >&2; exit 1; }
[[ -d "$code/third_party/openwam" ]] || { echo "missing vendored source in $code" >&2; exit 1; }

if [[ -f "$prefix/pip.conf" ]] && grep -q 'socks5' "$prefix/pip.conf"; then
  mv "$prefix/pip.conf" "$work/pip.conf.socks.failed"
fi

export PATH="$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DS_BUILD_OPS=0 CUDA_VISIBLE_DEVICES=
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export PIP_CACHE_DIR="$work/pip-cache" TMPDIR="$work/tmp" PIP_PROXY="$proxy"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
trap 'status=$?; printf "%s\n" "$status" > "$work/install-recover.exit"' EXIT

python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple \
  'pip>=25,<27' setuptools wheel packaging ninja
python -m pip install --report "$work/pip-torch.json" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128
python -m pip install --no-build-isolation --report "$work/pip-openwam.json" \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --constraint "$code/third_party/openwam/docker/constraints-cu128.txt" \
  "$code/third_party/openwam"
python -m pip check
python -m pip freeze > "$work/requirements.lock.txt"
python -m pip inspect > "$work/pip-inspect.json"
printf 'INSTALL_COMPLETE\n'
