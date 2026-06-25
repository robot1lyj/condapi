# Mode: Deployment

## Defaults
- Training server: `linyongjia@172.31.11.108` (SSH port 12222)
- Remote code: `/share/home/linyongjia/conda-pi/openpi`
- Remote datasets: `/share/home/linyongjia/datasets/`
- Remote output: `/share/home/linyongjia/output/openpi`
- Conda env: `pi-conda`
- Remote cache: `OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi`

## Build Offline Bundle (本地)
```bash
bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle
tar -C artifacts -cf pi-conda-offline-bundle.tar pi-conda-offline-bundle
scp -P 12222 pi-conda-offline-bundle.tar linyongjia@172.31.11.108:/share/home/linyongjia/
```

## Install on Server
```bash
ssh -p 12222 linyongjia@172.31.11.108
cd /share/home/linyongjia
tar -xf pi-conda-offline-bundle.tar
cd /share/home/linyongjia/conda-pi/openpi
bash scripts/conda/install_offline_bundle.sh \
  --bundle-dir /share/home/linyongjia/pi-conda-offline-bundle \
  --openpi-dir /share/home/linyongjia/conda-pi/openpi \
  --env-name pi-conda
```

## Sync Code
代码通过 `git push/pull` 在 `conda-pi` 分支同步。本地的 commit 推到远端后，在服务器 `git pull` 即可。

## Runtime (on Server)
```bash
# 环境变量
unset WANDB_DISABLED
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1 HUGGINGFACE_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi
export HF_LEROBOT_HOME=/share/home/linyongjia/datasets

# 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/train.py pi05_piper_dual \
  --exp-name <name> \
  --checkpoint-base-dir /share/home/linyongjia/output/openpi \
  --data.repo_id /share/home/linyongjia/datasets/<dataset>
```

## Serve (推理服务)
```bash
# ⚠️ --port 必须在 policy:checkpoint 之前
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/serve_policy.py --port 6666 policy:checkpoint \
  --policy.config=<config> --policy.dir=<dir>
```

## HQ 数据集训练 (1200 集, high_quality_folding)

### 1. 分割 + 正则化
```bash
conda run -n pi-conda python scripts/split_and_norm_relative.py \
    --dataset /share/home/linyongjia/datasets/high_quality_folding \
    --train-episodes 1000 --val-episodes 200 \
    --config pi05_openarms_dual_hq
```

### 2. 桥接 norm_stats 路径 (⚠️ 必须!)
```bash
mkdir -p assets/pi05_openarms_dual_hq
ln -sf /share/home/linyongjia/datasets/high_quality_folding \
    assets/pi05_openarms_dual_hq/high_quality_folding
```
不做这一步训练会报 `Norm stats not found`。

### 3. 训练
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/train.py pi05_openarms_dual_hq \
    --exp-name openarms_hq_bs32_fsdp2 \
    --checkpoint-base-dir /share/home/linyongjia/output/openpi
```
预估: ~76h (3.2天) on 2×A800, batch=32, 100k steps.

### 4. 离线评估
```bash
conda run -n pi-conda python scripts/evaluate_checkpoint.py \
    --config pi05_openarms_dual_hq \
    --checkpoint-dir <checkpoint_path> \
    --dataset /share/home/linyongjia/datasets/high_quality_folding \
    --val-split "1000:1200" \
    --output ./eval_report
```

## Pre-flight Checks
- 确认服务器可达: `ssh -p 12222 linyongjia@172.31.11.108 echo ok`
- 确认 conda 环境存在: `conda run -n pi-conda python -c "import openpi"`
- 确认 GPU 可用: `conda run -n pi-conda python -c "import jax; print(jax.devices())"`
