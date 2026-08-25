# 02 · 服务器与环境

本页是远端训练/验证服务器的唯一操作 owner：说明机器、目录、数据预检、单机 probe、多节点 K-Policy、监控和 checkpoint 交接。数据语义看 docs/04_data_contracts.md，训练策略看 docs/03_training_and_evaluation.md，服务与真机看 docs/05_inference_and_rollout.md。

下列节点、路径和端口是交接记录中的固定 profile，不等于当前实时占用或服务状态；每次操作都必须重新执行第 2、3 节的版本/GPU/tmux 检查。

## 1. 服务器拓扑与固定目录

~~~text
本地工作站
  -> jump host: ssh -p 12222 linyongjia@172.31.11.100
      -> GPU 节点: gpu12 / gpu14 / gpu18 / gpu19 / gpu25 / gpu28
~~~

gpu19 可能需要有效的 host key 和 Slurm allocation；PAM 拒绝时视为不可用。gpu25 是当前服务/采集候选，未经明确授权不得停止它的 6666 服务、启动第二个同端口服务或抢占 GPU。GPU 节点必须先做占用审计，不能凭主机名假设空闲。

| 节点 | 交接角色 | 使用前确认 |
|---|---|---|
| `gpu12` | 多节点 JAX coordinator 首选 | 必须是 `--hosts` 的第一个节点，确认 12365 未被占用 |
| `gpu14`、`gpu28` | 多节点训练候选 | 与 gpu12 组成完整作业；不能只启动其中一个 |
| `gpu18` | 评分/训练候选 | 先查实时 GPU/会话，不把历史空闲记录当现状 |
| `gpu19` | 受限候选 | host key、PAM 和 Slurm allocation 均通过后才使用 |
| `gpu25` | 服务/真机采集候选 | 默认端口 6666；未经授权不得停止、抢占或重启 |

远端共享路径：

~~~text
REPO_ROOT=/share/home/linyongjia/conda-pi/openpi
PYTHON=/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python
DATA_ROOT=/share/home/linyongjia/datasets
REFERENCE_ROOT=/share/home/linyongjia/data
OUTPUT_ROOT=/share/home/linyongjia/output/openpi
OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi
HF_HOME=/share/home/linyongjia/.cache/huggingface
~~~

进入 GPU 节点后先执行：

~~~bash
export REPO_ROOT=/share/home/linyongjia/conda-pi/openpi
export PYTHON=/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python
export DATA_ROOT=/share/home/linyongjia/datasets
export REFERENCE_ROOT=/share/home/linyongjia/data
export OUTPUT_ROOT=/share/home/linyongjia/output/openpi
export OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi
export HF_HOME=/share/home/linyongjia/.cache/huggingface
export HF_LEROBOT_HOME="$DATA_ROOT"
export WANDB_MODE=offline
export WANDB_SILENT=true
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO_ROOT"
~~~

训练输出约定为 $OUTPUT_ROOT/<CONFIG>/<EXP_NAME>/<STEP>；运行日志约定为 $OUTPUT_ROOT/logs/<CONFIG>/。KAI0 总控状态位于 $OUTPUT_ROOT/logs/openarm_kai0_pipeline_v1/status.json。

## 2. 首次登录与版本/环境预检

从本地进入服务器：

~~~bash
ssh -p 12222 linyongjia@172.31.11.100
ssh gpu12
~~~

在 GPU 节点执行；这些命令只读，不会修改远端代码：

~~~bash
cd "$REPO_ROOT"
git branch --show-current
git rev-parse --short HEAD
git status --short
test -x "$PYTHON"
"$PYTHON" -c "import openpi, torch; print('openpi ok; cuda=', torch.cuda.is_available())"
git submodule status
~~~

如果 git status 有未提交修改，不要直接 pull 或覆盖；先记录修改归属并确认远端代码 commit 与待运行 checkpoint/config 匹配。共享服务器上的代码同步由项目负责人决定，文档不假设某个远端 branch 自动跟随本地工作树。

## 3. GPU、进程和 tmux 审计

每次训练/服务前都执行一次：

~~~bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
tmux ls 2>/dev/null || true
pgrep -af '(serve_policy|scripts/train.py|monitor_openarm)' || true
ss -ltnp 2>/dev/null | grep ':6666' || true
~~~

低利用率不代表空闲：可能正在模型加载、JAX 编译、视频解码或异步保存。结合 GPU 进程、日志 mtime、CPU 和 tmux 再判断。一个多节点 JAX 作业是不可拆分单元，不能只重启一台节点。

## 4. 本地 conda 环境

本次交接基于 conda-pi 分支；联网机器构建离线包，再在目标机安装：

~~~bash
git submodule update --init --recursive
bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle
bash scripts/conda/install_offline_bundle.sh \
  --bundle-dir artifacts/pi-conda-offline-bundle --env-name pi-conda
conda run -n pi-conda python scripts/conda/patch_transformers.py --openpi-dir .
conda run -n pi-conda pip install --no-build-isolation --no-deps -e .
conda run -n pi-conda pip install --no-build-isolation --no-deps -e packages/openpi-client
~~~

远端运行时统一使用 pi-conda、离线 W&B/Hugging Face 和 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9；不要用 uv，不要在 jump host 上跑 GPU 训练，不要把 token 写入脚本。

## 5. 数据到训练的完整服务器闭环

下面是可以复核的顺序。DATASET、CONFIG、EXP_NAME 必须替换为同一实验的实体；不要把 Site probe 的默认值套到 KAI0。

### 5.1 服务器数据目录与容量检查

服务器上的数据按“原始/转换/冻结派生/训练输出”分层；不要把 `$REFERENCE_ROOT` 的 Stage/reference 目录当作 policy repo，也不要把 `/tmp` 中的 raw 当作可恢复的唯一副本。开始转换或训练前先确认空间、权限和数据入口：

~~~bash
export DATASET_ROOT="$DATA_ROOT/openarm_kai0_awbc_v1"
df -h "$DATA_ROOT" "$OUTPUT_ROOT"
du -sh "$DATA_ROOT" "$OUTPUT_ROOT" 2>/dev/null
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

### 5.3 单机 Site probe（新实验）

pi05_openarms_dual_site_align_v1_probe 的默认数据是 openarm_site_align_v1_deg，训练 split 为 0:141，不是正式 K-Data。新实验可在单节点 tmux 内执行：

~~~bash
export DATASET_ROOT="$DATA_ROOT/openarm_site_align_v1_deg"
export CONFIG=pi05_openarms_dual_site_align_v1_probe
export EXP_NAME=site_probe_DATE
export TMUX_NAME="train_$EXP_NAME"
mkdir -p "$OUTPUT_ROOT/logs/$CONFIG"
tmux new-session -s "$TMUX_NAME" -c "$REPO_ROOT"
~~~

进入 tmux 后逐行执行，完成后按 Ctrl-b d 脱离：

~~~bash
"$PYTHON" scripts/compute_openarm_parquet_norm_stats.py \
  --dataset "$DATASET_ROOT" --episodes 0:141
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "$PYTHON" scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" --checkpoint-base-dir "$OUTPUT_ROOT" \
  --batch-size 32 --num-workers 2 --num-train-steps 5000 --log-interval 20 \
  --overwrite 2>&1 | tee "$OUTPUT_ROOT/logs/$CONFIG/$EXP_NAME"_gpu12.log
~~~

监控：

~~~bash
tmux attach -t "$TMUX_NAME"
tail -f "$OUTPUT_ROOT/logs/$CONFIG/$EXP_NAME"_gpu12.log
~~~

### 5.4 正式 K-Policy 多节点

正式配置是 pi05_openarm_kai0_awbc_v1。当前记录的四卡拓扑为 gpu12 + gpu28、global batch 128、每进程 workers 8；launcher 默认拓扑却是 gpu12 + gpu14 + gpu28、六卡 batch 126，不能混淆。四卡 smoke：

~~~bash
"$PYTHON" scripts/launch_openarm_jax_multinode.py \
  --hosts gpu12 gpu28 --coordinator-address 172.31.11.112:12365 \
  --config pi05_openarm_kai0_awbc_v1 \
  --exp-name kpolicy_DATE_smoke20 \
  --num-train-steps 20 --batch-size 128 --num-workers 0 \
  --mode overwrite --session-prefix kpolicy_smoke \
  --xla-memory-fraction 0.90
~~~

smoke 产生完整 step 后才启动正式训练；新实验使用新 EXP_NAME，已有实验继续使用完全相同拓扑和 --mode resume：

~~~bash
"$PYTHON" scripts/launch_openarm_jax_multinode.py \
  --hosts gpu12 gpu28 --coordinator-address 172.31.11.112:12365 \
  --config pi05_openarm_kai0_awbc_v1 \
  --exp-name kpolicy_DATE_80k \
  --num-train-steps 80000 --batch-size 128 --num-workers 8 \
  --mode overwrite --session-prefix kpolicy_80k \
  --xla-memory-fraction 0.90
~~~

batch 必须能被 2 × 主机数整除；overwrite 只用于全新目录，已有目录只能在确认 checkpoint 完整后 resume。launcher 会在每个节点创建 tmux、设置离线缓存/JAX 分布式变量并写 $OUTPUT_ROOT/logs/<CONFIG>/ 日志。

### 5.5 KAI0 总控和监控

KAI0 总控 monitor_openarm_kai0_pipeline.py 是 controller，不是纯查看器；它可能推进评分、训练、sweep、部署和报告。只有项目负责人确认状态机和空闲节点后才运行：

~~~bash
export OPENPI_K_TRAIN_HOSTS=gpu12,gpu28
export OPENPI_K_GLOBAL_BATCH_SIZE=128
export OPENPI_K_NUM_WORKERS=8
export OPENPI_K_TRAIN_TAG=4gpu_gpu12_gpu28
"$PYTHON" scripts/monitor_openarm_kai0_pipeline.py --once
~~~

被动查看不启动 controller：

~~~bash
"$PYTHON" -m json.tool "$OUTPUT_ROOT/logs/openarm_kai0_pipeline_v1/status.json"
tail -f "$OUTPUT_ROOT/logs/openarm_kai0_pipeline_v1/history.jsonl"
tmux ls 2>/dev/null || true
~~~

多节点训练、评分或服务出现异常时，先保存 status/log/tmux/GPU 证据，再决定恢复；不要只 kill 一台节点。

## 6. checkpoint 交接

JAX checkpoint 只有下列条件同时满足才可服务：

~~~bash
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

- 不提交凭据，不删除数据/权重/cache，不停止 gpu25:6666；服务和采集先确认急停、人工接管、动作单位和工作空间。
- 端口监听、进程存在、训练 loss 下降都不是部署成功；必须完成 docs/05_inference_and_rollout.md 的真实 WebSocket smoke。
- 看到 OOM、视频尾帧、JAX 初始化、SSH 参数边界或 checkpoint 半写入问题时，保留原日志和状态，写入 docs/07_change_log.md，不要用删除缓存掩盖原因。
