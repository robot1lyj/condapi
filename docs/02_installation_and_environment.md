# 02 · 服务器与环境

本页是远端训练/验证服务器的唯一操作 owner：说明机器、目录、数据预检、单机 probe、多节点 K-Policy、监控和 checkpoint 交接。数据语义看 docs/04_data_contracts.md，训练策略看 docs/03_training_and_evaluation.md，服务与真机看 docs/05_inference_and_rollout.md。

下列新平台信息于 2026-09-04 按用户提供的《琶洲模方智算平台用户操作手册》（2026-04-15）和 SSH/Slurm 只读核验建立。作业、数据下载进度和资源占用都是接管快照；每次操作仍必须重新执行第 2、3 节检查。

## 1. 新平台入口与固定目录

~~~text
工作台网页： http://10.18.31.233:3080/
SSH：       wuyan@10.18.31.234:22
登录主机：  rocky-login.hlink.local
账号家目录：/home/wuyan
项目工作区：/home/wuyan/lyj/YAM
~~~

网页入口按平台手册要求使用可访问内网的 HTTP 代理；网页和 SSH 是两个入口，不能把 `.233:3080` 当作 SSH 地址。密码由用户/密码管理器提供，不写入仓库、脚本或命令历史。

| 变量/目录 | 当前值 | 接管时核验 |
|---|---|---|
| `HOME_ROOT` | `/home/wuyan` | 存在，登录用户为 `wuyan` |
| `PROJECT_ROOT` | `/home/wuyan/lyj/YAM` | 存在 |
| `CODE_ROOT` | `/home/wuyan/lyj/YAM/YAM_code` | 存在但接管时为空，尚未同步 OpenPI |
| `DATA_ROOT` | `/home/wuyan/lyj/YAM/YAM_data` | 存在，接管时约 23 GB |
| `ENV_PREFIX` | `/home/wuyan/.conda/envs/yam` | 存在；需先加载 `miniconda3/26.1.1` |
| `OUTPUT_ROOT` | 未确认 | 不得沿用旧 `/share/home/...` 路径，须以新作业实际输出目录为准 |

接管时已发现数据目录 `/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks`，并有 `train`/`val`、metadata、parquet 和视频下载日志；它尚未通过 OpenArm 16D/单位/任务合同审计，不能仅凭目录名接入 OpenArm 训练。

迁移后的 shell 可先设置已核实的目录变量；`REPO_ROOT` 和 `OUTPUT_ROOT` 在新平台实际同步/提交后再设置：

~~~bash
export HOME_ROOT=/home/wuyan
export PROJECT_ROOT=/home/wuyan/lyj/YAM
export CODE_ROOT="$PROJECT_ROOT/YAM_code"
export DATA_ROOT="$PROJECT_ROOT/YAM_data"
export ENV_PREFIX=/home/wuyan/.conda/envs/yam
~~~

平台手册说明工作台提供文件管理、作业编排、命令行、Jupyter、GUI 和运行总览；GUI/Jupyter 的 GPU、CPU 和时长以作业提交页的实时资源为准。不要把登录节点当 GPU 训练节点。

## 可调参数

参数只在一个地方改；改完必须重新跑对应 gate。表中 K-Policy 的步数/保存频率是旧项目配置参考，不代表新平台已有训练状态；新实验优先用 CLI 覆盖，不要修改正式配置或 launcher 默认值。

| 参数 | 修改入口 | K-Policy 当前值 | 修改后必须做什么 |
|---|---|---:|---|
| 数据集路径、输出路径 | 本页环境变量 `DATA_ROOT`、待确认的 `OUTPUT_ROOT` | 见 §1 | 检查目录、权限和磁盘空间 |
| 训练数据、split、norm 资产 | `src/openpi/training/config.py` 的独立 config | K-Data、`0:1719` | 新数据版本重算 norm，跑数据 audit/loader smoke |
| 计算节点、分区、coordinator | 新平台作业编排/Slurm；多节点 launcher 尚未迁移 | 接管时仅核实 `gpu` 分区上的 `gpu001` 下载作业 | 先查 `squeue`/`sinfo`；未核实前不得套用旧节点名或 coordinator |
| 全局 batch、workers | 新作业脚本或 launcher | 未确认 | 先在新平台单节点分配中做 smoke，再确定多节点拓扑 |
| 训练步数、保存频率 | config 或 launcher 的 `--num-train-steps`；保存频率在 config | `80000`、`5000` | 新 `EXP_NAME` 用 overwrite；续训只用 resume |
| XLA 显存比例 | launcher `--xla-memory-fraction` 或单机环境变量 | `0.90` | 记录显存和 OOM；不要用删 cache 规避问题 |
| norm 统计 episode | `compute_openarm_parquet_norm_stats.py --episodes` | `0:1719` | 必须与训练 split 完全一致，并写入同一数据版本 |
| 总控覆盖项 | 新平台迁移后的 launcher/controller | 未迁移 | 未完成代码、环境、数据和输出路径迁移前不得启动 controller |

模型 action 维度、单位、horizon 属于 [04 · 数据合同](04_data_contracts.md)；服务端端口、prompt、RTC 和 action-chunk 属于 [05 · 推理与 rollout](05_inference_and_rollout.md)。不要在本页复制它们的操作命令。

## 2. 首次登录与版本/环境预检

从本地进入服务器：

~~~bash
ssh -p 22 wuyan@10.18.31.234
~~~

登录后先加载平台环境；这些命令只读，不会修改远端代码：

~~~bash
module load miniconda3/26.1.1
conda activate /home/wuyan/.conda/envs/yam
hostname
id
pwd
echo "$CONDA_PREFIX"
python --version
test -d /home/wuyan/lyj/YAM/YAM_code
test -d /home/wuyan/lyj/YAM/YAM_data
~~~

接管时上述环境激活后 Python 为 3.13.12；在登录节点导入 `torch` 失败，尚未证明它适合 OpenPI。仓库要求 Python 3.11，必须在新平台实际 GPU 作业内完成依赖/import gate 后，才能安装或运行 OpenPI。`YAM_code` 当前为空，不要执行 `git pull`、覆盖目录或把本地仓库自动同步上去。

## 3. GPU、进程和 tmux 审计

每次训练/服务前都执行一次；先在工作台或 Slurm 确认分配，再到计算节点执行 GPU 检查：

~~~bash
squeue -u "$USER"
sinfo
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
tmux ls 2>/dev/null || true
pgrep -af '(serve_policy|scripts/train.py|monitor_openarm)' || true
~~~

低利用率不代表空闲：可能正在模型加载、JAX 编译、视频解码或异步保存。结合 GPU 进程、日志 mtime、CPU、Slurm 状态和 tmux 再判断。一个多节点 JAX 作业是不可拆分单元，不能只重启一台节点。

## 4. 新平台 Conda 环境

新平台不是旧的 `pi-conda` profile。当前可复核的环境入口是平台 module 加用户环境：

~~~bash
module load miniconda3/26.1.1
conda activate /home/wuyan/.conda/envs/yam
export PYTHON="$CONDA_PREFIX/bin/python"
python --version
~~~

`yam` 环境接管时为 Python 3.13.12，登录节点未安装 `torch`；它目前只能作为待审计的现有环境，不能直接宣称满足 OpenPI。待 `CODE_ROOT` 同步仓库且在 GPU 作业内通过 Python 3.11/`jax`/`torch`/`lerobot`/`openpi_client` import gate 后，才决定是否复制平台公共环境或在该环境中安装项目依赖。不要使用 uv，不要在登录节点跑 GPU 训练，不要把 token 写入脚本。

## 5. 数据到训练的完整服务器闭环

下面是可以复核的顺序。DATASET、CONFIG、EXP_NAME 必须替换为同一实验的实体；不要把接管时的 `ABC-130k-two-tasks` 目录直接套到 OpenArm，也不要把旧服务器的 KAI0 快照当作新平台现状。

### 5.1 服务器数据目录与容量检查

服务器上的数据按“原始/转换/冻结派生/训练输出”分层；新平台的 reference/output 根目录尚未确认，不要自行创建兼容旧服务器的别名，也不要把 `/tmp` 中的 raw 当作可恢复的唯一副本。开始转换或训练前先确认空间、权限和数据入口：

~~~bash
export DATASET_ROOT="$DATA_ROOT/openarm_kai0_awbc_v1"
df -h "$DATA_ROOT"
du -sh "$DATA_ROOT" 2>/dev/null
test -d "$DATASET_ROOT" || { echo "OpenArm K-Data 未在新平台核实：$DATASET_ROOT" >&2; exit 1; }
test -d "$DATASET_ROOT/meta" && test -d "$DATASET_ROOT/data" && test -d "$DATASET_ROOT/videos"
find "$DATASET_ROOT/meta" -maxdepth 1 -type f -printf '%f\n' | sort
find "$DATASET_ROOT/data" -type f -name '*.parquet' | wc -l
find "$DATASET_ROOT/videos" -type f | wc -l
"$PYTHON" -m json.tool "$DATASET_ROOT/meta/info.json" | sed -n '1,100p'
~~~

`meta/info.json`、`episodes.jsonl`、parquet、videos、`norm_stats.json` 和 build report 必须来自同一冻结版本；文件数量只能作为发现异常的线索，最终以 5.2 的正式 audit 为准。磁盘满、目录不可读、软链接指向临时目录或 metadata 的 `data_path/video_path` 不可解析时，停止流程并先修复数据版本。

### 5.2 数据和 norm 预检

先看 docs/04_data_contracts.md 的数据矩阵和单位，再在 GPU 节点执行：

~~~bash
export DATASET_ROOT="$DATA_ROOT/openarm_kai0_awbc_v1"
test -f "$DATASET_ROOT/meta/info.json"
test -f "$DATASET_ROOT/meta/episodes.jsonl"
test -f "$DATASET_ROOT/meta/episodes_stats.jsonl"
test -f "$DATASET_ROOT/meta/tasks.jsonl"
test -f "$DATASET_ROOT/kai0_awbc_build_report.json"
~~~

正式 K-Data 缺少 episodes_stats.jsonl、build report 或视频时应停止，不要在训练前静默修复冻结数据。只有在另存新版本并获得授权后，才可使用 scripts/write_lerobot_episode_stats.py 或 scripts/repair_lerobot_subset.py 生成派生结果。

norm stats 必须和训练 episode 一致；正式 K-Data 只在冻结版本上计算一次：

~~~bash
if [[ ! -f "$DATASET_ROOT/norm_stats.json" ]]; then
  "$PYTHON" scripts/compute_openarm_parquet_norm_stats.py \
    --dataset "$DATASET_ROOT" --episodes 0:1719
fi
~~~

正式 K-Data 的结构和真实 OpenPI loader gate：

~~~bash
mkdir -p "$OUTPUT_ROOT/logs/openarm_kai0_pipeline_v1"
"$PYTHON" scripts/audit_openarm_kai0_training_data.py \
  --dataset "$DATASET_ROOT" \
  --config pi05_openarm_kai0_awbc_v1 \
  --expected-episodes 1719 \
  --expected-hq 999 --expected-site 420 --expected-tda 300 \
  --output "$OUTPUT_ROOT/logs/openarm_kai0_pipeline_v1/k_data_training_audit.json"
"$PYTHON" scripts/openarm_benchmark_loader.py \
  pi05_openarm_kai0_awbc_v1 --backend torchcodec \
  --num-workers 0 --batch-size 2 --batches 2
~~~

audit 会检查二值 task、来源数量、连续 episode/frame/index、norm stats、真实 16D loader、三类视频样本和每个 TDA episode 尾帧；它通过后才允许启动正式训练。

### 5.3 新平台单机作业入口（待迁移）

平台手册给出的默认入口是“作业编排”页面，也可以从命令行进入 Slurm。当前 `CODE_ROOT` 为空、`OUTPUT_ROOT` 未确认，且 `yam` 还未通过 OpenPI 依赖 gate，因此本节暂不启动训练。迁移完成后必须在新作业中先做 20-step/loader smoke，并将代码、配置、数据、norm 和输出目录绑定到同一份记录。

~~~bash
module load miniconda3/26.1.1
conda activate /home/wuyan/.conda/envs/yam
sinfo
squeue -u "$USER"
tmux ls 2>/dev/null || true
~~~

长任务仍使用 Slurm 作业或 tmux；提交前确认作业脚本目录、`--chdir`、标准输出目录和资源申请。不要复用旧服务器的 `gpu12/gpu14/gpu28`、coordinator 或 `pi-conda` 命令，除非已完成新平台适配并重新核验。

### 5.4 正式 K-Policy 迁移闸门

`pi05_openarm_kai0_awbc_v1`、K-Data 1719 集和旧多节点拓扑属于上一台训练服务器的项目快照。新平台当前只核实到 `ABC-130k-two-tasks` 下载任务，尚未核实 OpenArm K-Data、OpenPI 仓库、checkpoint、输出根目录或多节点 coordinator；因此不能在新平台直接启动正式 K-Policy。

迁移顺序固定为：同步完整仓库到新 `CODE_ROOT` → 在计算节点核验 Python 3.11/依赖 → 确认 OpenArm 数据与 norm → 单节点 loader/20-step smoke → 核实 Slurm 多节点资源 → 才决定是否迁移 launcher 和正式训练。任何一步失败都保留日志，不删除数据或缓存掩盖问题。

### 5.5 新平台任务监控

被动查看使用 Slurm 和实际作业日志，不调用尚未迁移的 KAI0 controller：

~~~bash
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode,NodeList
tmux ls 2>/dev/null || true
tail -f /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/download.log
~~~

接管时作业 `1962/abc-download` 在 `gpu001` 的 `gpu` 分区运行，标准输出与错误均指向该数据目录的 `slurm-1962.out`；这是正在进行的下载任务，不是 OpenPI 服务。异常时先保存 `squeue`/`sacct`/日志/GPU 证据，不要只 kill 一台节点。

## 6. checkpoint 交接

JAX checkpoint 只有下列条件同时满足才可服务：

~~~bash
export OUTPUT_ROOT=replace_with_new_platform_output_root
export CONFIG=pi05_openarm_kai0_awbc_v1
export EXP_NAME=replace_with_exp_name
export STEP=replace_with_step
export CHECKPOINT="$OUTPUT_ROOT/$CONFIG/$EXP_NAME/$STEP"
test -f "$CHECKPOINT/_CHECKPOINT_METADATA"
test -f "$CHECKPOINT/params/_METADATA"
test -f "$CHECKPOINT/assets/openarm_kai0_awbc_v1/norm_stats.json"
~~~

部署前同时记录 config、dataset、训练 commit、step、prompt、norm stats 路径和日志；数字目录或只有 params/ 的异步中间态不可晋级。`replace_with_exp_name` 和 `replace_with_step` 必须替换成真实值。policy server 会优先从 checkpoint 的 assets/<asset_id>/norm_stats.json 加载 stats，因此不能用另一数据版本的 stats 替换它。

## 7. 服务器安全与失败处理

- 不提交凭据，不删除数据/权重/cache，不停止接管时的 `1962/abc-download`；服务和采集先确认急停、人工接管、动作单位和工作空间。
- 端口监听、进程存在、训练 loss 下降都不是部署成功；必须完成 docs/05_inference_and_rollout.md 的真实 WebSocket smoke。
- 看到 OOM、视频尾帧、JAX 初始化、SSH 参数边界或 checkpoint 半写入问题时，保留原日志和状态，写入 docs/07_change_log.md，不要用删除缓存掩盖原因。
