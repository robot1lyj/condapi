#!/usr/bin/env bash
# Bounded post-install verification; never starts training or changes raw data.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: $0 ENV_PREFIX TEST_SNAPSHOT RAW_ROOT INSTALL_LOG" >&2
  exit 2
fi
prefix=$1
snapshot=$2
raw_root=$3
install_log=$4
deadline=$((SECONDS + 86400))
while ! grep -qx 'INSTALL_COMPLETE' "$install_log"; do
  if ! tmux has-session -t condapi-env-install 2>/dev/null; then
    # Recheck after session exit to avoid a completion/exit race.
    grep -qx 'INSTALL_COMPLETE' "$install_log" && break
    echo 'INSTALL_FAILED: inspect installation log; no automatic overwrite' >&2
    exit 1
  fi
  [[ $SECONDS -lt $deadline ]] || { echo 'INSTALL_WAIT_TIMEOUT' >&2; exit 1; }
  echo "WAIT_INSTALL $(date -Is)"
  sleep 30
done
export PATH="$prefix/bin:$PATH"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset PYTHONPATH
cd "$snapshot"
run_dir=$(mktemp -d "$snapshot/run.XXXXXXXX")
echo "VERIFICATION_RUN=$run_dir"
sha256sum scripts/convert_yam_subset.py scripts/convert_yam_subset_test.py \
  scripts/audit_yam_subset.py scripts/audit_yam_subset_test.py
"$prefix/bin/python" -m pip check
# Isolated synthetic fixtures only. The real dataset is never an output target.
"$prefix/bin/python" -m pytest -q scripts/convert_yam_subset_test.py scripts/audit_yam_subset_test.py \
  --basetemp "$run_dir/fixtures"
echo 'SYNTHETIC_CONVERSION_VERIFIED'
"$prefix/bin/python" scripts/audit_yam_subset.py "$raw_root" --inventory-only > "$run_dir/inventory.json"
"$prefix/bin/python" scripts/audit_yam_subset.py "$raw_root" --limit 1 > "$run_dir/structure-sample.json"
"$prefix/bin/python" - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

for name in ('inventory.json', 'structure-sample.json'):
    report = json.loads((Path(sys.argv[1]) / name).read_text())
    print(name, json.dumps({k: v for k, v in report.items() if k != 'episodes'}, ensure_ascii=False))
print('VERIFICATION_COMPLETE; REAL_CONVERSION_AND_GPU_NOT_VERIFIED')
PY
