#!/bin/bash
# Executed inside launch_offline.sbatch after immutable inputs are published.
set -euo pipefail
: "${PLD_RUN_ROOT:?}"
source "$PLD_RUN_ROOT/run.env"
cd "$PLD_CODE_ROOT"
mkdir -p "$PLD_RUN_ROOT/configs"
"$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/audit_replay.py --replay "$PLD_REPLAY_ROOT" --output "$PLD_RUN_ROOT/replay-audit.json"
"$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/gpu_checks.py --output "$PLD_RUN_ROOT/gpu-checks.json"
"$PLD_ENV_PREFIX/bin/python" -m pip freeze > "$PLD_RUN_ROOT/environment-freeze.txt"
IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ ${#devices[@]} != 4 ]]; then
    echo 'Expected an allocation with four GPUs' >&2
    exit 2
fi
CUDA_VISIBLE_DEVICES="${devices[0]}" "$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/cache_features.py \
    --replay "$PLD_REPLAY_ROOT" --raw-root "$PLD_RAW_ROOT" --weights "$PLD_ENCODER_WEIGHTS" \
    > "$PLD_RUN_ROOT/feature-cache.log" 2>&1
"$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/register_dataset.py \
    --project-root "$PLD_PROJECT_ROOT" --replay "$PLD_REPLAY_ROOT" --raw-root "$PLD_RAW_ROOT" \
    --classification "$PLD_CLASSIFICATION_ROOT"
ids=(calql-success-s0 calql-success-s1 calql-mixed-s0 td-success-s0)
letters=(a b c d)
pids=()
for index in 0 1 2 3; do
    cp "configs/pld_rl/offline14-${letters[$index]}.json" "$PLD_RUN_ROOT/configs/${ids[$index]}.json"
    CUDA_VISIBLE_DEVICES="${devices[$index]}" "$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/train_offline.py \
        --replay "$PLD_REPLAY_ROOT" --config "$PLD_RUN_ROOT/configs/${ids[$index]}.json" \
        --output "$PLD_RUN_ROOT/${ids[$index]}" > "$PLD_RUN_ROOT/${ids[$index]}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
"$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/summarize_runs.py --run-root "$PLD_RUN_ROOT"
exit "$failed"
