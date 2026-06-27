# Mode: Deployment

## Defaults
- Training access: jump host `linyongjia@172.31.11.122:12222`
- Current training node: `linyongjia@172.31.11.108`
- Remote code: `/share/home/linyongjia/conda-pi/openpi`
- Remote data root: `/share/home/linyongjia/data`
- Remote output: `/share/home/linyongjia/output/openpi`
- Conda env: `pi-conda`
- Remote cache: `OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi`
- Default dataset addressing: `/share/home/linyongjia/data/local/<alias> -> ../<dataset>` and `--data.repo_id local/<alias>`.

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
export HF_LEROBOT_HOME=/share/home/linyongjia/data

# 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/train.py pi05_piper_dual \
  --exp-name <name> \
  --checkpoint-base-dir /share/home/linyongjia/output/openpi \
  --data.repo_id local/<alias>
```

## Remote Train Helper
Use the shared helper for norm-stats-then-train runs:

```bash
bash ~/.codex/skills/openpi-conda-remote-train/scripts/start_remote_train.sh \
  --node 172.31.11.108 \
  --dataset-name <dataset_dir_under_data> \
  --repo-alias <alias> \
  --config pi05_piper_dual \
  --exp-name <name>
```

## Serve (推理服务)
```bash
# ⚠️ --port 必须在 policy:checkpoint 之前
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/serve_policy.py --port 6666 policy:checkpoint \
  --policy.config=<config> --policy.dir=<dir>
```

## Legacy Absolute-Path Training
Older docs and OpenArm configs may use `/share/home/linyongjia/datasets/<dataset>` directly. Treat that as compatibility context unless the user explicitly asks for that path.

## Pre-flight Checks
- 确认节点可达: `ssh -J linyongjia@172.31.11.122:12222 -p 12222 linyongjia@172.31.11.108 echo ok`
- 确认 conda 环境存在: `conda run -n pi-conda python -c "import openpi"`
- 确认 GPU 可用: `conda run -n pi-conda python -c "import jax; print(jax.devices())"`
