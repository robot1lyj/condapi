#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/piper_dataset_train.sh --dataset-dir DIR --exp-name NAME [options]

Required:
  --dataset-dir DIR          Versioned dataset directory containing meta/data/videos
  --exp-name NAME            Experiment name

Optional:
  --config NAME              Train config (default: pi05_piper_dual)
  --checkpoint-base-dir DIR  Output root (default: /share/home/linyongjia/output/openpi)
  --batch-size N             Batch size (default: 32)
  --fsdp-devices N           FSDP devices (default: 2)
  --max-frames N             If set, recompute norm stats using at most N frames
  --overwrite                Pass --overwrite to train.py
  --resume                   Pass --resume to train.py
  --skip-norm                Skip compute_norm_stats.py
EOF
}

DATASET_DIR=""
EXP_NAME=""
CONFIG_NAME="pi05_piper_dual"
CHECKPOINT_BASE_DIR="/share/home/linyongjia/output/openpi"
BATCH_SIZE="32"
FSDP_DEVICES="2"
MAX_FRAMES=""
OVERWRITE=0
RESUME=0
SKIP_NORM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --exp-name)
      EXP_NAME="$2"
      shift 2
      ;;
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --checkpoint-base-dir)
      CHECKPOINT_BASE_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --fsdp-devices)
      FSDP_DEVICES="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --skip-norm)
      SKIP_NORM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET_DIR" || -z "$EXP_NAME" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "$DATASET_DIR/meta/info.json" ]]; then
  echo "Dataset metadata not found: $DATASET_DIR/meta/info.json" >&2
  exit 1
fi

if [[ $SKIP_NORM -eq 0 ]]; then
  NORM_CMD=(
    python scripts/compute_norm_stats.py
    --config-name "$CONFIG_NAME"
    --repo-id "$DATASET_DIR"
  )
  if [[ -n "$MAX_FRAMES" ]]; then
    NORM_CMD+=(--max-frames "$MAX_FRAMES")
  fi
  "${NORM_CMD[@]}"
fi

TRAIN_CMD=(
  python scripts/train.py "$CONFIG_NAME"
  --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  --data.repo_id "$DATASET_DIR"
  --batch-size "$BATCH_SIZE"
  --fsdp-devices "$FSDP_DEVICES"
)

if [[ $OVERWRITE -eq 1 ]]; then
  TRAIN_CMD+=(--overwrite)
fi
if [[ $RESUME -eq 1 ]]; then
  TRAIN_CMD+=(--resume)
fi

"${TRAIN_CMD[@]}"
