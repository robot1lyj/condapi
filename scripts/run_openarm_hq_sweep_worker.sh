#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu-index> <checkpoint-step> [checkpoint-step ...]" >&2
  exit 2
fi

GPU_INDEX="$1"
shift

REPO_DIR="${REPO_DIR:-/share/home/linyongjia/conda-pi/openpi}"
CONDA_BIN="${CONDA_BIN:-/share/home/linyongjia/miniconda3/bin/conda}"
DATASET="${DATASET:-/share/home/linyongjia/datasets/high_quality_folding}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/share/home/linyongjia/output/openpi/eval/hq_checkpoint_sweep_20260706}"
LOG_DIR="${LOG_DIR:-/share/home/linyongjia/output/openpi/logs/eval/hq_checkpoint_sweep_20260706}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

for STEP in "$@"; do
  OUT="${OUTPUT_ROOT}/ckpt_${STEP}"
  LOG="${LOG_DIR}/ckpt_${STEP}_gpu${GPU_INDEX}.log"
  rm -rf "${OUT}"
  mkdir -p "${OUT}"
  {
    date
    "${CONDA_BIN}" run --no-capture-output -n pi-conda \
      python scripts/evaluate_openarm_checkpoint_sweep.py \
        --config pi05_openarms_dual_hq \
        --dataset "${DATASET}" \
        --checkpoint "${CHECKPOINT_ROOT}/${STEP}" \
        --train-max-episodes 120 \
        --val-max-episodes 200 \
        --uniform-frames 3 \
        --critical-frames 3 \
        --output "${OUT}"
    date
  } >"${LOG}" 2>&1
  echo "${STEP} done" >>"${LOG_DIR}/gpu${GPU_INDEX}.done"
done
