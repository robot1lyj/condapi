#!/bin/bash
# Four allocated GPUs, two waves; no robot or VLA training entry is invoked.
set -euo pipefail
: "${PLD_RUN_ROOT:?}"
source "$PLD_RUN_ROOT/run.env"
cd "$PLD_CODE_ROOT"
mkdir -p "$PLD_RUN_ROOT/configs"
"$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/audit_replay.py --replay "$PLD_REPLAY_ROOT" --output "$PLD_RUN_ROOT/replay-audit.json"
"$PLD_ENV_PREFIX/bin/python" -m pip freeze > "$PLD_RUN_ROOT/environment-freeze.txt"
"$PLD_ENV_PREFIX/bin/python" -m pip check > "$PLD_RUN_ROOT/pip-check.txt"
"$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/method_gpu_checks.py --output "$PLD_RUN_ROOT/gpu-checks.json"
IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[[ ${#devices[@]} == 4 ]] || exit 2
recipes=(pld-iql-s0 pld-iql-s1 pld-td3bc-s0 pld-rebrac-s0 pld-iql-e09-s0 pld-iql-e09-s1 pld-td3bc-s1 pld-rebrac-s1)
failed=0
for wave in 0 1; do
    pids=()
    for index in 0 1 2 3; do
        recipe=${recipes[$((wave*4+index))]}
        config="configs/pld_rl/methods14-$recipe.json"
        id=$("$PLD_ENV_PREFIX/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$config")
        [[ "$id" =~ ^[a-z0-9-]+$ ]] || exit 2
        cp "$config" "$PLD_RUN_ROOT/configs/$id.json"
        CUDA_VISIBLE_DEVICES="${devices[$index]}" "$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/train_methods.py \
            --replay "$PLD_REPLAY_ROOT" --config "$PLD_RUN_ROOT/configs/$id.json" \
            --output "$PLD_RUN_ROOT/$id" > "$PLD_RUN_ROOT/$id.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
done
"$PLD_ENV_PREFIX/bin/python" scripts/pld_rl/summarize_runs.py --run-root "$PLD_RUN_ROOT" --phase offline-actor-critic-comparison
exit "$failed"
