# 耳机装配 Pi0.5 50h 配置准备（2026-10-12）

当前授权（2026-10-13）：用户明确要求开始训练，第一阶段累计目标 **200,000步**，允许直接使用服务器。已将 first_stop_step 改为200000，保留原338000步学习率衰减计划；本阶段到200000停止，不自动推进到338000。数据准备及启动正在执行，实际作业、就绪状态与首步证据见后续运行记录。2026-10-12完成的配置与选集准备保留如下。

## 配方

配置：[pi05-earbuds-50h.json](../../../../configs/native/pi05-earbuds-50h.json)。任务为 `insert the wireless bluetooth earbuds into the charging case`，即将蓝牙耳机放入充电盒；使用数据集 task 作为 prompt，不将子任务语言自动改作逐帧训练指令。

| 项目 | 固定设置 |
|---|---|
| 模型 / 初始化 | `pi05_yam` 全量微调；官方 `pi05_base/params`，新优化器；不继承乐高任务权重 |
| 训练子集 | seed42 从原 train 随机选完整轨迹，1,384 集、5,402,194 帧、30fps、50.020315h |
| 验证集 | 原 val 的全部17集、62,173帧、0.575676h；不参与抽样、梯度或 norm |
| 动作 | 原始14D absolute；训练关节 delta、夹爪 absolute，mask `(6,-1,6,-1)`；模型内部32D，H50 |
| RTC | training-time RTC，最大延迟10（d=0..10）；旧普通训练/推理路径保留，默认延迟0仍关闭RTC |
| 资源 | global batch32、4卡FSDP、64 CPU、480G、单次6天；不代表已获得空闲资源 |
| 优化器 / LR | Adam eps1e-6、EMA关闭、seed42；peak1e-5、末端1e-6、warmup1000、cosine decay固定338000 |
| 训练预算 | ceil(5,402,194×2/32)=337,638步；总338,000，约2.00215轮；本次累计200,000；338,000仅保留为原总预算，不自动进入后续阶段 |
| 保存 | 每1,000步保存，长期保留每5,000步及最新；续训保持相同代码/数据/norm/优化器/LR合同 |

“50h”按单条轨迹的30fps时长计量，三路相机不重复计时；等效轮数按训练帧起点计算，不乘H50。约一轮/两轮是训练预算，不是任务成功率承诺。已有乐高步时不是耳机实测工期。

## 数据证据与剩余准备

2026-10-12 10:10:56 +08 经 `yam-server` 只读核验；没有修改源数据、下载任务、模型环境或现有作业。完整证据在 [server-audit.json](server-audit.json)。

- 源 train：2,095集、8,233,457帧、76.235713h；固定 revision `68651e4929d9fb00f798937b2d62617cab5c771d`。val revision `46ca817c39d3df08115a75c31b3cd3867e3d875a`。
- 所选 train 和全部 val 的 **5,464,367帧**均检查了14D state/action shape、有限值、完整帧号和30fps时间戳，通过；训练涉及536个源Parquet文件，val涉及1个。
- train 的4,152个视频、val的51个视频均非空；每split首/中/末三条轨迹的三相机共18个首帧解码成功，224×224、AV1。**本次不是全视频逐帧验收，也不是逐集任务成功审核。**
- 选中 train 视频约51.924 GB，val约0.587 GB；独立拷贝另需至少这些空间以及Parquet/元数据/norm空间。观察时共享home余量约5.0T，开工前仍须复核。
- 夹爪 action 均在[0,1]；state少量高于1（最大1.0001812），保留原值，不裁剪。单位依据归 [contract.json](contract.json)：发布者弧度/夹爪0闭1开和固定port顺序，raw到port单位继承仍是文档加数值支持的推断，未宣称独立标定。
- `source-train.jsonl`、`source-val.jsonl` 保留来源manifest原始字节；[selection.json](selection.json)保留选集与来源哈希。[source-episodes.json](source-episodes.json)是原train源ID；[episodes.json](episodes.json)是新转换目录的连续ID 0..1383，二者不可互换。
- **尚无已发布的耳机LeRobot转换目录及耳机专属norm。** 不能复用乐高norm或把 `RECIPE_READY_DATA_UNVERIFIED` 当成训练数据READY。新数据版本将独立copy视频、检查SHA256、全帧解码、审核低维列、验收真实LeRobot loader；随后用所选train全部帧计算H50/delta统计，并验证真实loader等价。val不进入统计。

源目录：`/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/earbuds_charging_case`。
新转换目录：`/home/wuyan/lyj/YAM/YAM_data/processed/earbuds_lerobot_50h_v1_20261012/{train,val}`。
资产目录：`/home/wuyan/lyj/YAM/training-assets/earbuds_rtc_50h_20261012`。
新run：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_20261012`。
控制目录：`/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_20261012`。

## 执行入口与授权边界

本地仅执行标准库配方检查和轻量配置测试，没有训练循环或模型前反向：

```bash
python3 scripts/pi05_earbuds_50h.py check --recipe-only
```

服务器代码仅从Gitea取得不可变checkout；开训前按当前授权同步固定代码，不更改现有模型环境。下列变量由操作方设置为实际Gitea checkout及其完整commit，不使用占位路径启动。CPU转换/全视频审计也只在计算节点执行；prepare-data没有训练调用或后继作业提交：

```bash
export EARBUDS_CODE_DIR=/absolute/path/to/immutable/gitea-checkout
export EARBUDS_CODE_COMMIT=FULL_40_CHARACTER_COMMIT
sbatch "$EARBUDS_CODE_DIR/scripts/prepare_pi05_earbuds_50h.sbatch"
```

断点转换支持同一命令追加 `--resume`，核对合同与源签名后复用已验收copy；不会删除原件。不完整norm目录保留供诊断，拒绝覆盖。准备完成后，在服务器执行以下检查，只有实际输出 `DATA_READY` 后才具备此配置的数据前置条件：

```bash
/home/wuyan/.conda/envs/condapi-yam/bin/python \
  "$EARBUDS_CODE_DIR/scripts/pi05_earbuds_50h.py" check
```

**2026-10-13用户已授权首次训练到200000；以下首次命令用于本次执行。** 启动脚本要求Slurm、固定代码commit、干净的相关源码、独立已验收数据与norm；数据准备不调用训练脚本。普通 `check` 不写文件也不构造模型。

```bash
sbatch "$EARBUDS_CODE_DIR/scripts/train_pi05_earbuds_50h.sbatch" --stop-step 200000
# 同一已授权阶段中断，核对旧作业已退出且checkpoint完整后恢复到累计200000：
sbatch "$EARBUDS_CODE_DIR/scripts/train_pi05_earbuds_50h.sbatch" --resume --stop-step 200000
```

正式训练复用 `scripts/train_lego_full.py` 和OpenPI训练器；新增显式 `--dataset` 参数必须同时给出初始化权重、选集、专属norm，验证norm的数据路径及转换manifest/info哈希；未给该参数时乐高旧默认保持原行为。首轮错误退出且没有完整checkpoint时保留现场、诊断后新建run，不删除合同绕过保护。没有自动续训、自动提交后继作业、Thor部署或客户端任务消息。

## 验证

轻量配置/防误启测试24项通过；覆盖乐高旧默认、耳机dataset/norm错配、源metadata变化、任务资产绑定、累计停止步数、固定LR、无Slurm拒绝训练、数据未准备拒绝训练，以及recipe-only不执行子进程。Ruff、两份sbatch的Bash语法和 `git diff --check` 通过。未执行转换或真实GPU训练验证；这些完成前本页仅表示配置准备就绪。
