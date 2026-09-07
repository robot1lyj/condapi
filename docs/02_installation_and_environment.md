# 02 · 服务器与 Thor 环境

本页是训练服务器、conda、数据预检和训练资源的操作 owner，也记录 Thor 从零安装的入口。YAM 数据语义见 [04 · 数据合同](04_data_contracts.md)，Pi0.5 端侧转换和验收见 [08 · Thor 端侧部署](08_thor_edge_deployment.md)，训练入口见 [03 · 训练与评估](03_training_and_evaluation.md)。

服务器信息于 2026-09-04 按用户提供的《琶洲模方智算平台用户操作手册》（本地参考文件：`/home/wuyan-lyj/下载/琶洲模方智算平台用户操作手册_带目录.pdf`）以及 SSH/Slurm 只读核验建立；Thor 官方版本和端侧方案于 2026-09-05 更新。一次性状态每次操作前都要重新检查。

## Thor 端侧系统与从零安装

本节是环境概览，不是安全烧录操作清单。实际指导用户从 [Thor 安装系列入口](reference/thor/00_start_here.md) 开始，按阶段核验设备、USB、NVMe 与固件；不可跳过其中的目标确认和停止条件。

当前端侧目标暂按 Jetson AGX Thor Developer Kit（T5000 口径）记录；设备到手后先核对实际 SKU。官方最新系统基线是 JetPack 7.2.1 / Jetson Linux r39.2.1 / Ubuntu 24.04 / Kernel 6.8 / CUDA 13.2.1 / TensorRT 10.16.2。完整部署决策和 Pi0.5 调研见 [08 · Thor 端侧部署](08_thor_edge_deployment.md)。

本次已从 NVIDIA 官方地址下载 Jetson ISO，原始文件放在仓库外；已核对文件长度、类型并记录本地摘要：

```text
/home/wuyan-lyj/thor-system/jetpack-7.2.1/jetsoninstaller-r39.2.1-2026-08-07-18-30-47-arm64.iso
```

下载观察时间、文件大小、本地 SHA-256 与官方参考校验的证据边界见 [08 · 官方系统基线](08_thor_edge_deployment.md#2-官方系统基线)。

制作系统盘前必须完成：

1. 若重新下载或替换 ISO，必须重新执行 `sha256sum`，并同步更新 `docs/08_thor_edge_deployment.md` 和 `docs/07_change_log.md`。
2. 在至少 25 GB 可用空间的主机上，用 Balena Etcher 将 ISO 写入至少 16 GB U 盘；ISO 不能作为 Live USB 直接试运行。
3. Thor 从 U 盘启动，若出现 QSPI capsule update 提示确认 `Y`，安装目标选择 `Install on NVMe`；安装完成拔出 U 盘，再完成 `oem-config`。
4. 首次启动在 Thor 上记录 `cat /etc/nv_tegra_release`、`uname -a`、`jetson_release`、`docker --version`、`dpkg-query -W nvidia-container-toolkit` 和 `nvidia-smi`。

Jetson ISO 安装方式通常已经带 Docker 和 NVIDIA Container Toolkit；若使用 SDK Manager 或 `Linux_for_Tegra` 刷写，按 [Thor Docker Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html) 补装，不要无审计地覆盖容器运行时。Thor 按模型系列隔离容器，Pi 系列共用一个服务，首版在其中验证原生 JAX checkpoint；宿主机维护系统、驱动、Docker/Compose/Toolkit，模型依赖在系列镜像构建阶段安装。镜像固定、挂载、端口与切换约定见 [08 · 容器方案](08_thor_edge_deployment.md#31-按模型系列隔离容器的部署约定)，官方基镜像口径见该页第 3.2 节；系统级开发组件仅在确有需要时从 JetPack 源安装，禁止安装 Ubuntu 的 `nvidia-cuda-toolkit`。

系统版本与 GPU 环境 gate 通过后，才把本仓库和 checkpoint 放到 Thor；随后按 [08](08_thor_edge_deployment.md) 的原 JAX golden、LoRA/FP32 转换审计和未量化后端对照完成 YAM 验收，再决定是否需要 TensorRT 或量化。系统盘刷写、Docker smoke 和 YAM engine 目前都不能仅凭文档宣称已验证。

## 1. 服务器和目录

```text
SSH：          wuyan@10.18.31.234:22
登录主机：     rocky-login.hlink.local
工作台：       http://10.18.31.233:3080/
服务器家目录： /home/wuyan
项目工作区：   /home/wuyan/lyj/YAM
```

| 变量 | 当前路径/状态 | 说明 |
|---|---|---|
| `PROJECT_ROOT` | `/home/wuyan/lyj/YAM` | 服务器工作区 |
| `CODE_ROOT` | `/home/wuyan/lyj/YAM/YAM_code` | 当前项目 `condapi` 的目标目录；不放独立 YAM-ABC 代码 |
| `DATA_ROOT` | `/home/wuyan/lyj/YAM/YAM_data` | 已存在的数据目录 |
| `RAW_YAM_DATA` | `/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks` | 已发现，需先做 LeRobot/合同审计 |
| 项目环境 | `/home/wuyan/.conda/envs/condapi-yam` | Python 3.12；待在服务器创建/安装 |
| 输出根目录 | 未确认 | 不沿用旧服务器路径，首次 Slurm 作业时确定 |

服务器数据下载任务 `1962/abc-download` 在 `gpu001` 上运行时不可停止、删除或抢占；本项目只读取其完成后的数据，不改变下载进程。

服务器登录后使用：

```bash
export PROJECT_ROOT=/home/wuyan/lyj/YAM
export CODE_ROOT="$PROJECT_ROOT/YAM_code"
export DATA_ROOT="$PROJECT_ROOT/YAM_data"
export RAW_YAM_DATA="$DATA_ROOT/ABC-130k-two-tasks"
```

密码只通过交互式 SSH/密码管理器提供，绝不写入仓库、脚本、remote URL 或日志。

## 2. 环境方案

### 2.1 创建项目环境

本仓库的 `environment.pi-conda.yml` 已收敛为 Python 3.12；服务器 conda 已核实可以使用 conda-forge 镜像，pip 经大 wheel 实测选择腾讯云 PyPI 镜像。创建前确认 `CODE_ROOT` 是当前 `condapi` 仓库，而不是 `/home/wuyan-lyj/YAM/yam-abc-reproduce`：

```bash
module load miniconda3/26.1.1
conda env create -f "$CODE_ROOT/environment.pi-conda.yml" \
  -p /home/wuyan/.conda/envs/condapi-yam
conda activate /home/wuyan/.conda/envs/condapi-yam
export PYTHON="$CONDA_PREFIX/bin/python"
python --version
```

如果项目环境已存在，只允许先检查版本和包，再决定是否补装；不删除环境或清理其他用户缓存。

### 2.2 安装当前仓库

先安装项目声明的完整依赖，再以 editable 方式覆盖本地 `openpi` 和 `openpi-client`。不使用 uv；本地测速选用腾讯云镜像：

```bash
cd "$CODE_ROOT"
"$PYTHON" -m pip install --index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
  --upgrade-strategy only-if-needed -e .
"$PYTHON" -m pip install --no-build-isolation --no-deps -e packages/openpi-client
```

依赖安装完成后，在 GPU 计算节点而不是登录节点执行 import gate：

```bash
"$PYTHON" - <<'PY'
import jax
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import openpi
import openpi.policies.yam_policy as yam_policy
import torch

print("jax", jax.__version__)
print("lerobot", getattr(lerobot_dataset, "__file__", "<unknown>"))
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("yam_action_dim", yam_policy.YAM_STATE_ACTION_DIM)
PY
```

完整 import 或 CUDA gate 失败时，停止训练启动，记录具体版本/错误；不要通过把 YAM-ABC 代码复制进本项目来绕过依赖问题。

### 2.3 本地验证环境与当前模型资产

服务器尚未恢复可用前，先使用本地 Miniconda 环境完成代码和资产验证：

```bash
export LOCAL_PYTHON=/home/wuyan-lyj/miniconda3/envs/condapi-yam/bin/python
export OPENPI_DATA_HOME=/home/wuyan-lyj/.cache/openpi
"$LOCAL_PYTHON" --version
"$LOCAL_PYTHON" -m pip check
```

当前 Pi0.5 LoRA 路线只需要 Pi0.5 基础 checkpoint 和 PaliGemma tokenizer；两项已在本地缓存，分别为：

```text
$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params/
$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model
```

基础 checkpoint 目录必须保留 `commit_success.txt` 后才能认为下载完整。Pi0、Pi0-Fast 或 FAST tokenizer
不是当前 `pi05_yam_lora` 的必需资产，除非后续明确新增对应实验，不提前下载。YAM 的 `assets/yam/norm_stats.json`
仍必须在正式数据版本审计后计算，不能用 Pi0.5 基础资产或其他机器人合同的统计量代替。

## 3. Slurm 和 GPU 预检

登录节点只用于提交/查看任务。每次创建作业先查看资源，GPU 型号和显存必须在实际分配后记录：

```bash
squeue -u "$USER"
sinfo
scontrol show partition gpu
srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty bash
```

进入计算节点后：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
"$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)'
```

长训练用 Slurm batch/job 或 tmux 保持会话；不要在 login shell 前台运行训练。训练前检查 `squeue`，不得影响 `1962/abc-download`。

## 4. 数据接入前检查

`RAW_YAM_DATA` 的目录名不等于可训练的 LeRobot repo。先确认数据是否已经是当前 LeRobot v3 布局，尤其是 `meta/info.json`、tasks metadata、parquet、三路视频和 feature names：

```bash
test -d "$RAW_YAM_DATA"
find "$RAW_YAM_DATA" -maxdepth 2 -type f | sort | sed -n '1,80p'
find "$RAW_YAM_DATA" -name 'info.json' -o -name 'tasks.parquet' -o -name 'episodes*.jsonl' | sort
df -h "$DATA_ROOT"
```

只有通过 [04 · 数据合同](04_data_contracts.md) 的键、14D 形状、task/prompt、视频首中尾解码和 norm gate 后，才能将 `--data.repo-id` 指向它。若需转换，必须写到新目标目录，原始下载目录保持只读。

## 5. 代码同步和远端备份

当前本地仓库是 `condapi`；服务器 `YAM_code` 只接收本仓库的代码。不得同步 `/home/wuyan-lyj/YAM/yam-abc-reproduce` 到该目录，也不得把它的控制/GUI 依赖作为训练依赖。

本地工作站提交后使用两个远端：Gitea 是服务器唯一代码来源，GitHub 只由本地工作站维护备份。服务器无法连接 GitHub，不在服务器配置或执行 GitHub push/pull：

```bash
git push -u origin main    # 本地 → Gitea
git push github main       # 本地 → GitHub
```

服务器端如需通过 Git 同步，只使用 Gitea `origin` 的 clone/fetch/pull；先确认服务器公钥已登记到 Gitea，再使用非交互认证。Gitea Git 身份为 `wuyan_lyj <linyongjia@wuyanai.cn>`；未登记前不要把密码拼进 URL。服务器代码同步、环境安装和数据审计必须分开记录。

用于登记 Gitea 的服务器侧公钥（此前从服务器环境读取并核实）是下面这一把；它不是本机连接服务器所用的
`/home/wuyan-lyj/.ssh/id_ed25519_condapi_yam.pub`：

```text
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDvsyWnnsxG4PskVm36sn9WVFp1TiWyFBq92zN8RP0gTMLxxloN96LokVb2HSymzQ0yfrzxd2/RKX7ykjYgWhE3VVbjaoIBDDOTlxVisr/3G8LGIFxgd+KmSsmhnMRjNGwSnoqrAUJf3c87D9/zFnlLP1g9+p7B5wYXXfX8rzsdcJICKEFosrC1OQMKOB9kRK+e+NCvQwe0Th2O92rVcny4jI728k3IEITtzaMh1U5udKP7K9nCwaqOUhXRFzodjZKzuxuQb7kSlBt9y8wmbyOEancDU4WCQc0N6xm5b5kt9ynUcPkxPAmB4BKIMOiOOPA4dEcZBCMbDTveIHJK8Dn1AeJG9HTgHaSyPlgUIZLLYN3lg0aieG5qSlOE/nTQYaLM3zKpIyAqXC6YSoxKMpQ4XcowJaky89JuuQEpAoSe2drD2XNwdAfdYBM7m+SXge2riHqQ3g7mO4PMsR9yNJ0op0Vd0aIXDCmrWi5eWO6vZ/aTWsAhxQB9m/UIOrUL0= root@rocky-ood.hlink.local
```

## 6. 安全边界

- 不提交或展示私钥、密码、token；服务器只读公钥可交给用户登记 Gitea。
- 不删除远端数据、权重、缓存、日志或现有下载任务。
- 不在登录节点训练或进行需要 GPU 的 import/benchmark。
- 新环境失败时保留错误日志，不删除其他环境或清理其他用户缓存。
