# 02 · 环境与安装

## 本地 conda 环境

本次交接基于 `conda-pi` 分支；开始训练或合并前先运行 `git branch --show-current`，确认没有落到上游默认分支。

项目目标为 Python 3.11，GPU 训练通常需要 CUDA 12。联网机器构建离线包，再把包带到训练机：

```bash
git submodule update --init --recursive
bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle
bash scripts/conda/install_offline_bundle.sh \
  --bundle-dir artifacts/pi-conda-offline-bundle --env-name pi-conda
conda run -n pi-conda python scripts/conda/patch_transformers.py --openpi-dir .
conda run -n pi-conda pip install --no-build-isolation --no-deps -e .
conda run -n pi-conda pip install --no-build-isolation --no-deps -e packages/openpi-client
```

如果只做 CPU 单元测试，可让 `conftest.py` 自动选择 CPU；训练和真实推理不要据此判断 GPU 环境可用。

## 远端资源

默认通过跳板进入 GPU 节点：

```text
ssh -p 12222 linyongjia@172.31.11.100
ssh gpu12       # 训练节点示例，实际先查占用
```

交接记录中的路径为：

```text
repo:    /share/home/linyongjia/conda-pi/openpi
env:     /share/home/linyongjia/miniconda3/envs/pi-conda
data:    /share/home/linyongjia/datasets
output:  /share/home/linyongjia/output/openpi
cache:   /share/home/linyongjia/.cache/openpi
```

另有少量 Stage/reference 数据位于 `/share/home/linyongjia/data`。不要将它当作新 policy 数据集根目录；启动前检查配置中实际的 `repo_id`、`assets` 和 `norm_stats` 路径。

## 训练服务器原则

- 长训练、评分和服务放入 tmux；记录 session 名、节点、配置、checkpoint 和日志路径。
- 使用 `pi-conda`，不要在 jump host 上执行 GPU 训练，也不要用 uv 代替 conda。
- 多节点 JAX 作业是一个不可拆分单元；先做同拓扑 smoke，再启动正式任务。
- 训练/服务使用离线友好的 W&B/Hugging Face 设置；不把 token 或私有凭据写入脚本和文档。
- GPU 节点变更、远端路径变更属于 hot fact，必须同步 `docs/cache/kernel.md`。

## 环境变量

```bash
export OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

按任务覆盖，不要把本地 `/home/...` 路径写进远端 config。需要远端自动训练时，优先使用 `openpi-conda-remote-train` skill 的脚本，再人工确认配置和数据版本。

## 交接检查

```bash
conda run -n pi-conda python -c "import openpi, torch; print('openpi ok', torch.cuda.is_available())"
git submodule status
git status --short
```

安装完成不代表数据或服务可用；继续执行 [训练与评估](03_training_and_evaluation.md) 的 norm/loader smoke，以及 [推理与 rollout](05_inference_and_rollout.md) 的真实 WebSocket smoke。
