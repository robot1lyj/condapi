#!/usr/bin/env bash
# Install a verified conda-pack archive into a new prefix only.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: $0 ARCHIVE SHA256 NEW_PREFIX CODE_ROOT" >&2
  exit 2
fi
archive=$1
expected=$2
prefix=$3
code_root=$4
wheel_dir="$(dirname "$archive")/repair-wheels"
[[ -f "$wheel_dir/packaging-25.0-py3-none-any.whl" && -f "$wheel_dir/setuptools-80.10.2-py3-none-any.whl" ]]
[[ "$prefix" == /* && "$prefix" != / && "$prefix" != /home && "$prefix" != /home/wuyan ]]
[[ -f "$archive" && -f "$code_root/pyproject.toml" ]]
[[ "$expected" =~ ^[a-f0-9]{64}$ ]]
actual=$(sha256sum "$archive")
[[ "${actual%% *}" == "$expected" ]] || { echo 'Archive checksum mismatch' >&2; exit 1; }
[[ ! -e "$prefix" ]] || { echo "Refusing to overwrite existing prefix: $prefix" >&2; exit 1; }
mkdir -p "$(dirname "$prefix")"
mkdir "$prefix"
tar -xzf "$archive" -C "$prefix"
export PATH="$prefix/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
git -C "$code_root" rev-parse HEAD
"$prefix/bin/python" "$prefix/bin/conda-unpack"
# conda-pack may restore cached conda files over pip replacements. Repair offline
# before loading the editable-build backend, which imports packaging/setuptools.
"$prefix/bin/python" -m pip install --no-index --no-deps --force-reinstall \
  "$wheel_dir/packaging-25.0-py3-none-any.whl" "$wheel_dir/setuptools-80.10.2-py3-none-any.whl"
"$prefix/bin/python" -m pip install --no-index --no-deps --no-build-isolation -e "$code_root"
"$prefix/bin/python" -m pip install --no-index --no-deps --no-build-isolation -e "$code_root/packages/openpi-client"
"$prefix/bin/python" -m pip check
# CPU-only ABI/import checks; GPU availability is a separate Slurm gate.
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$prefix/bin/python" - <<'PY'
import importlib
from importlib.metadata import version
import platform
print('Python:', platform.python_version())
for name in ['numpy', 'scipy', 'pyarrow', 'av', 'torch', 'jax', 'lerobot', 'openpi']:
    module = importlib.import_module(name)
    print(name, version(name), module.__file__)
from openpi.policies.yam_policy import YAM_STATE_ACTION_DIM
assert YAM_STATE_ACTION_DIM == 14
print('CPU_ENV_VERIFIED; GPU_GATE_PENDING', flush=True)
PY
echo 'INSTALL_COMPLETE'
