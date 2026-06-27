# Piper 双臂离线 Conda 训练说明（服务器非容器环境）

这是 `conda-pi` 分支当前默认的训练路径。

这份说明对应 `conda-pi` 分支，目标是：

- 使用 `pi-conda` 作为唯一训练环境
- 不依赖训练容器内的项目虚拟环境
- 改成在训练服务器的非容器 `conda` 环境中直接运行 `openpi`
- 所有 Python 依赖先在本地下载好，再离线传到服务器安装

当前训练目标仍然是 `Piper` / `pi05_piper_dual` 这条链路。默认数据根目录为
`/share/home/linyongjia/data`，训练时使用 `local/<alias>` 指向具体版本目录；旧文档中的
`/share/home/linyongjia/datasets` 属于兼容路径。

## 1. 目录和脚本

本分支新增了这几份文件：

- `environment.pi-conda.yml`
- `scripts/conda/conda-specs-linux-64.txt`
- `scripts/conda/requirements-pi-pip.txt`
- `scripts/conda/build_offline_bundle.sh`
- `scripts/conda/install_offline_bundle.sh`
- `scripts/conda/patch_transformers.py`

其中：

- `build_offline_bundle.sh` 在本地联网机器执行
- `install_offline_bundle.sh` 在训练服务器非容器环境执行

## 2. 本地打离线包

在本地 `openpi` 仓库根目录执行：

```bash
bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle
```

输出目录结构：

```text
artifacts/pi-conda-offline-bundle/
├── conda_pkgs/
├── wheelhouse/
├── src/
└── meta/
```

说明：

- `conda_pkgs/` 是服务器创建 `conda` 环境所需的离线包
- `wheelhouse/` 是 `pip --no-index` 离线安装所需的 wheel
- `src/` 里放了 pin 住的 `lerobot` 和 `dlimp` 源码快照，对应 wheel 也在本地一起打好后再传服务器
- `meta/` 里是显式安装清单和版本记录

打包传输：

```bash
tar -C artifacts -cf pi-conda-offline-bundle.tar pi-conda-offline-bundle
scp -P 12222 pi-conda-offline-bundle.tar linyongjia@172.31.11.108:/share/home/linyongjia/
```

## 3. 服务器离线安装

在训练服务器非容器环境执行。默认代码目录使用我们现在同步过去的分支副本：
`/share/home/linyongjia/conda-pi/openpi`

```bash
cd /share/home/linyongjia
tar -xf pi-conda-offline-bundle.tar
cd /share/home/linyongjia/conda-pi/openpi

bash scripts/conda/install_offline_bundle.sh \
  --bundle-dir /share/home/linyongjia/pi-conda-offline-bundle \
  --openpi-dir /share/home/linyongjia/conda-pi/openpi \
  --env-name pi-conda
```

安装完成后，所有运行命令都改成：

```bash
conda run -n pi-conda python ...
```

## 4. 训练前环境变量

推荐固定这些环境变量：

```bash
unset WANDB_DISABLED
export WANDB_MODE=offline
export WANDB_SILENT=true
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi
export HF_HOME=/share/home/linyongjia/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_LEROBOT_HOME=/share/home/linyongjia/data
```

## 5. 训练命令

先算归一化统计。当前推荐为版本化数据集创建 `local/<alias>` 软链，再用 `repo_id=local/<alias>`：

```bash
mkdir -p /share/home/linyongjia/data/local
ln -sfn ../piper_dish_v001 /share/home/linyongjia/data/local/dish
```

```bash
conda run -n pi-conda python scripts/compute_norm_stats.py \
  --config-name pi05_piper_dual \
  --repo-id local/dish
```

再启动训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/train.py pi05_piper_dual \
  --exp-name piper_ft_pi05_dish \
  --checkpoint-base-dir /share/home/linyongjia/output/openpi \
  --data.repo_id local/dish \
  --log-interval 20
```

## 6. 离线训练曲线

训练默认使用离线 W&B，不会联网同步。每个实验目录下会同时生成：

- `wandb/`：W&B offline run 文件，可后续手动 sync 或用 W&B 工具查看。
- `metrics/metrics.jsonl`：逐次日志点的结构化指标。
- `metrics/metrics.csv`：同一批指标的表格版本。
- `metrics/plots/training_curves.png`：聚合训练曲线图。
- `metrics/plots/<metric>.png`：每个指标的单独曲线图。

`--log-interval 20` 表示每 20 step 记录一次，比默认 100 step 更密；如果需要更细可以设为 `10`。

## 7. transformers 补丁

`openpi` 当前仍然需要把 `src/openpi/models_pytorch/transformers_replace/` 覆盖到安装后的 `transformers` 里。

离线安装脚本已经自动做了这一步：

```bash
conda run -n pi-conda python scripts/conda/patch_transformers.py \
  --openpi-dir /share/home/linyongjia/conda-pi/openpi
```

如果后面重装了 `transformers`，再执行一次上面这条命令即可。
