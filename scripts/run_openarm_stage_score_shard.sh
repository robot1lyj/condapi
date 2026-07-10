#!/usr/bin/env bash

set -euo pipefail

GPU_ID=""
EPISODES=""
SHARD_NAME=""
BATCH_SIZE="32"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --shard-name)
            SHARD_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$GPU_ID" || -z "$EPISODES" || -z "$SHARD_NAME" ]]; then
    echo "Usage: $0 --gpu-id N --episodes START:END --shard-name NAME [--batch-size N]" >&2
    exit 2
fi

REPO_ROOT="/share/home/linyongjia/conda-pi/openpi"
PYTHON="/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python"
SOURCE="/share/home/linyongjia/datasets/high_quality_folding"
DESTINATION="/share/home/linyongjia/datasets/openarm_kai0_stage_scores_hq_v1_${SHARD_NAME}"
CHECKPOINT="/share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701/10000"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "$PYTHON" -u scripts/openarm_stage_advantage_awbc.py \
    --src "$SOURCE" \
    --dst "$DESTINATION" \
    --checkpoint "$CHECKPOINT" \
    --episodes "$EPISODES" \
    --task "Fold the T-shirt properly" \
    --batch-size "$BATCH_SIZE" \
    --device cuda:0 \
    --score-only \
    --copy-mode hardlink \
    --overwrite
