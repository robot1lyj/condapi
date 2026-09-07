#!/usr/bin/env bash
# Publish a new two-split version only after both independent conversions succeed.
set -euo pipefail
[[ $# == 5 || ( $# == 6 && $6 == --resume ) ]] || { echo 'Usage: ENV_PREFIX CODE_SNAPSHOT RAW_ROOT OUTPUT_VERSION CONTRACT [--resume]' >&2; exit 2; }
prefix=$1
snapshot=$2
raw_root=$3
version=$4
contract=$5
staging="$version.incomplete"
resume_args=()
[[ ${6:-} != --resume ]] || resume_args=(--resume)
export PATH="$prefix/bin:$PATH"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset PYTHONPATH
"$prefix/bin/python" - "$raw_root" "$version" "$staging" "${6:-}" <<'PY'
from pathlib import Path
import sys
raw, final, staging = map(lambda p: Path(p).resolve(), sys.argv[1:4])
if not raw.is_dir() or final == raw or raw in final.parents or final in raw.parents:
    raise ValueError('Output must be separate from the raw source tree')
if (final.exists() or staging.exists()) and sys.argv[4] != '--resume':
    raise FileExistsError('Use a new output version; no overwrite or automatic cleanup')
if final.exists() and staging.exists():
    raise ValueError('Both published and incomplete versions exist; inspect before resuming')
PY
[[ -f "$contract" && -f "$snapshot/scripts/convert_yam_subset.py" ]]
mkdir -p "$(dirname "$version")"
exec 9>>"$version.lock"
flock -n 9 || { echo 'Another conversion owns this version' >&2; exit 1; }
if [[ -d "$version" ]]; then
  staging="$version"
else
  mkdir -p "$staging"
fi
sha256sum "$snapshot/scripts/convert_yam_subset.py" "$snapshot/scripts/audit_yam_subset.py" "$contract"
for split in val train; do
  echo "START_SPLIT=$split $(date -Is)"
  "$prefix/bin/python" "$snapshot/scripts/convert_yam_subset.py" "$raw_root" "$staging/$split" \
    --split "$split" --contract "$contract" --video-mode copy "${resume_args[@]}"
  echo "SPLIT_VERIFIED=$split $(date -Is)"
done
if [[ "$staging" != "$version" ]]; then
  mv -T -n "$staging" "$version"
  [[ ! -e "$staging" ]] || { echo 'Publication refused: destination exists' >&2; exit 1; }
fi
echo "CONVERSION_COMPLETE=$version $(date -Is)"
