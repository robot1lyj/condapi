#!/bin/bash
#SBATCH --job-name=pld-offline14-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
: "${PLD_RUN_ROOT:?set the immutable experiment directory}"
mkdir -p "$PLD_RUN_ROOT"
date -Is
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
# Preparation runs on the login CPU. Never consume a partially published bundle.
deadline=$((SECONDS + 3600))
while [[ ! -f "$PLD_RUN_ROOT/READY" ]]; do
    if (( SECONDS > deadline )); then
        echo 'Preparation did not publish READY within one hour' >&2
        exit 2
    fi
    sleep 5
done
exec bash "$PLD_RUN_ROOT/run_pipeline.sh"
