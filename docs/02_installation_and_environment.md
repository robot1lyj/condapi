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

### MolmoAct2 / LeRobot 独立环境

2026-09-08接入官方LeRobot普通MolmoAct2，与Evo使用同一固定源码wheel，依赖分别安装。工作站prefix为 `/home/wuyan-lyj/.conda/envs/vla-molmoact2-dev`，服务器为 `/home/wuyan/.conda/envs/vla-molmoact2-train`。规格/版本/哈希锁在 `environments/molmoact2*`，profiles在 `configs/environments/molmoact2-*`。新环境专属动态库变量设为 `LD_LIBRARY_PATH=<prefix>/lib`，不修改系统或旧环境。

本地与服务器安装和CPU检查均已完成，两端108个Python包版本一致；实测与SSH中断后的补装记录见 [环境证据](reports/environments/molmoact2-20260908/README.md)。具体复现、LoRA/FP32动作专家、YAM双夹爪与分位数统计差异见 [模型接入说明](reference/molmoact2_integration.md)。不把本次环境安装记作真实GPU训练/Thor部署验收。

### Evo-1 / LeRobot 独立环境

2026-09-08实际结果：本地与服务器CPU检查均通过，106个Python包版本一致。服务器新环境通过专属 `LD_LIBRARY_PATH=<prefix>/lib` 使用Conda FFmpeg，已写入该环境的Conda环境变量；不修改系统库，直接运行prefix内Python时也要传入。证据见 [报告](reports/environments/evo1-20260908/README.md)。

Evo-1 不安装根目录 OpenPI 依赖，不克隆或更新正在训练的 `condapi-yam`。本地 prefix 为 `/home/wuyan-lyj/.conda/envs/vla-evo1-dev`，服务器 prefix 为 `/home/wuyan/.conda/envs/vla-evo1-train`。环境规格在 `environments/evo1.yml`，核心 Python 依赖和 LeRobot 固定提交在 `environments/evo1-requirements.txt`；本地、服务器 profile 分别为 `configs/environments/evo1-workstation.toml`、`configs/environments/evo1-server.toml`。

固定 LeRobot 源码 `2774d9bddcbbda50e697e162e89e7eaada8d7105`（包版本0.6.2），仅安装 evo1/training extras。Python3.12、FFmpeg7、Torch2.10.0、TorchVision0.25.0、TorchCodec0.10.0 为本次环境组合，不要求其他模型跟随。源码检出 `/home/wuyan-lyj/lerobot-evo1-2774d9b`；安装包在本地 `/home/wuyan-lyj/evo1-install-2774d9b`、服务器 `/home/wuyan/lyj/evo1-install-2774d9b`，均不放入项目 Git。

这套锁定包只适用于 Linux x86_64 / Python3.12，不是 Thor ARM 镜像。完整104个wheel版本、URL和SHA256在 `environments/evo1-wheels.lock.json`；包含从固定提交构建的 LeRobot wheel，不用 PyPI 上同版本号的其他构建替代。profile 的 lock 文件纳入运行计划哈希。

2026-09-08 实际安装中，服务器 pip 直连清华/阿里镜像有大包低速问题，但同一地址的 curl 可明显更快（cuDNN706MB实测约32秒）。使用 `scripts/conda/fetch_locked_wheels.py` 两并发下载并校验，再用 pip 离线安装。服务器本次wheel暂存 `/tmp/evo1-wheels-wuyan-2774d9b`，该目录可能被系统清理，不能作为持久权重目录；固定源码wheel和报告仍在家目录，锁文件随 Git 保存。

```bash
# 在服务器登录节点运行；只安装新 prefix，不进入既有 condapi-yam。
nice -n 15 /home/public/conda/miniforge3/bin/python \
  /home/wuyan/lyj/evo1-install-2774d9b/fetch_locked_wheels.py \
  /home/wuyan/lyj/evo1-install-2774d9b/evo1-wheels.lock.json \
  /tmp/evo1-wheels-wuyan-2774d9b \
  --source-wheel /home/wuyan/lyj/evo1-install-2774d9b/lerobot-0.6.2-py3-none-any.whl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 \
  /home/wuyan/.conda/envs/vla-evo1-train/bin/python -m pip install \
  --no-compile --no-index --find-links /tmp/evo1-wheels-wuyan-2774d9b \
  '/home/wuyan/lyj/evo1-install-2774d9b/lerobot-0.6.2-py3-none-any.whl[evo1,training]' \
  torch==2.10.0 torchvision==0.25.0 torchcodec==0.10.0
```

首次服务器 Conda 创建被中断后，离线重试发现该安装尝试缓存缺文件；没有清空或修复共享缓存。最终基础环境改用本地打包，哈希一致后解包到新prefix并运行 conda-unpack。后续需要 Conda 新包时优先用新的专用 package cache，或重建独立环境，不把这次不完整缓存误报为现有训练环境损坏。共享家目录小文件写入较慢，首次安装耗时不代表模型训练吞吐。

服务器训练存在时，只允许低优先级下载、安装和 CPU 检查；不抢占 GPU、不更新驱动，不进入原训练环境执行 pip install。新 prefix 安装失败时，保留/隔离该次不完整目录，不用清理共享缓存来解决问题。可用本地 `conda-pack` 基础环境迁移后执行 `conda-unpack`；安装包直传慢时，传固定源码 wheel，大依赖从镜像下载并校验哈希。

CPU 检查入口：在目标环境执行 `python scripts/conda/audit_evo1_environment.py --output <新建审计JSON>`。脚本强制隐藏 GPU、关闭 Hub 联网，只检查训练/Evo 模块导入、TorchCodec 动态库和合成14D/三相机 processor；不下载权重或初始化完整模型。检查通过不等于 GPU 前反向、真实 YAM 数据或训练效果已验收。实际安装结果与完整包清单归 `docs/reports/environments/evo1-20260908/`，模型 capability 仍由 [10](10_vla_platform.md) 区分。

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

### 本地环境压缩迁移（2026-09-07）

本地环境 `/home/wuyan-lyj/miniconda3/envs/condapi-yam` 约 8.5GB，已用 conda-pack 0.9.2 打包到
`/home/wuyan-lyj/condapi-env-transfer/condapi-yam-20260907.tar.gz`（约 4.2GB）。SHA-256：
`8005d7f2639cb31e3e67018cfbca08b93b33d76de99366497f600e3ecbe511ba`。
两端 x86_64，本地 glibc 2.39、服务器 glibc 2.34；1115 个 ELF 的符号需求静态检查未发现超过 2.34 的要求，
但这不能替代远端导入/CUDA 验收。

打包使用 `--ignore-editable-packages --ignore-missing-files`：后者仅在确认缺失项来自 conda 原始
setuptools/packaging 被 pip 固定版本替换后采用；实际运行版本是 setuptools 80.10.2、packaging 25.0，
本地 `pip check` 通过。独立前缀解压验证发现 conda-pack 会混入缓存中的原始 conda 文件，导致
`packaging._ranges` 缺失；安装器必须先从压缩包同级 `repair-wheels/` 离线强制重装这两个精确版本，
再安装 editable 包。修复 wheel 已同步服务器，本地独立前缀修复后 editable 安装、`pip check`、
CPU 导入和 24 项转换/数据加载相关测试通过；服务器实际运行验收仍待安装结束。
不得把忽略缺失文件选项作为未知错误的通用绕过方式。

`scripts/conda/migrate_local_env.sh` 是本次迁移入口，使用 flock 避免重复执行，rsync 断点续传、有限重试。
本地后台服务 `condapi-env-migration-20260907.service` 负责上传；日志在
`/home/wuyan-lyj/condapi-env-transfer/migration.log`。服务独立于对话，但上传期间本地电脑须保持开机和网络，
当前用户未启用 linger，注销本地用户也可能停止用户服务。

本次上传现已完成，远端 SHA-256 与上述值一致。原后台服务在传输结束后因运行中脚本被修改而疑似读取错位，
报语法错误，未自动启动安装；2026-09-07 已直接在服务器启动下述 tmux 安装流程，不重传、不覆盖旧环境。
当前处于共享盘解包阶段，尚无 `INSTALL_COMPLETE`；此后安装不再依赖本地电脑或连接。
后续不要修改正在执行的 shell 脚本；脚本更新应在进程结束后生效。

安装后的自动验证由远端 tmux `condapi-env-verify` 运行 `scripts/conda/verify_remote_env.sh`，
等待安装成功最多 24 小时；安装进程失败则停止，不覆盖目标环境。日志位于
`/home/wuyan/lyj/YAM/env-transfer/verification-b87e88a/verify.log`。
该目录仅放本次转换/审计测试快照，不更新 `YAM_code`，不会替代正式 Gitea 同步。
验证先运行合成三相机转换与回读测试，再对真实数据执行全量文件清点及每 split 首个 episode 的结构审计，
输出保存在独立 `run.*` 目录。`SYNTHETIC_CONVERSION_VERIFIED` 仅表示合成转换通过，
`VERIFICATION_COMPLETE` 也不代表真实数据单位已确认或 GPU/训练验收完成。

上传结束后自动启动远端 tmux `condapi-env-install`，执行 `scripts/conda/install_packed_env.sh`：
核对压缩包 SHA-256 → 拒绝覆盖既有目标 → 解压 → conda-unpack → 离线修复 packaging/setuptools →
离线重装本仓库和客户端 editable 包 →
pip check → CPU 导入检查。目标为 `/home/wuyan/.conda/envs/condapi-yam`，
日志 `/home/wuyan/lyj/YAM/env-transfer/install.log`；只有日志出现 `INSTALL_COMPLETE` 才能记录迁移成功。
失败时保留目录与日志，不能自动删除目标；GPU gate 仍等待实际资源分配和设备验收。

服务器直连内网 Gitea 曾超时，本次通过本地 SSH 反向隧道让服务器访问 Gitea，已 clone 到 `YAM_code`；
服务器 origin 保留正式 Gitea 地址，未配置 GitHub。隧道只服务代码同步，不参与环境安装或作业运行。

环境补充（2026-09-07）：共享盘正式安装仍在慢速解包。为继续 CPU 数据验证，另在登录节点本地盘
`/tmp/condapi-yam-smoke.CzQI6oAj/env` 完成同一压缩包的安装，pip check、CPU 导入及 9 项转换/审计测试通过。
这是临时 CPU 验证环境，不是共享训练环境；重启/临时目录清理可能使其失效，不供计算节点训练引用。
安装日志 `/tmp/condapi-yam-smoke.CzQI6oAj/install.log` 已出现 INSTALL_COMPLETE。
不能据此宣布 `/home/wuyan/.conda/envs/condapi-yam` 已完成，后者仍以自身安装日志为准。

## 3. Slurm 和 GPU 预检

最新复核（2026-09-07）：调度接口已恢复，按用户要求撤销仍在排队的作业 2063，
按原官方四卡模板重新提交的作业 **2064** 已受理；仅保留新申请。
申请 `gpu` 分区、单节点 4 GPU、64 CPU、480G、7 天，作业名 `yam-4gpu-workspace`。
当前 `PENDING / Priority`，预计启动时间 `N/A`；尚无分配节点，不能宣称获得 4090 或 CUDA 验收通过。
Slurm 独立持有该作业，不依赖 SSH/对话；运行后由现有有界 workspace 脚本保留资源，最长 7 天，不会自动续租。
只保留此一个申请，不重复提交。节点显示 IDLE，但 `sdiag` 部分统计时间比登录节点当前时间晚约 5 小时，
属于待平台核实的时间异常线索，不作为排队原因的确定结论。下文先前失败记录应视为历史观测。

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
