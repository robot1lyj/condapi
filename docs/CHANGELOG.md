# openpi Change Log

Record **what changed + why**. Not config values, verification steps, or impact scope —
those live in config files, git log, and CI respectively.

Design decisions → `docs/decisions/`. Architecture → `docs/ARCHITECTURE.md`.

---

## 2026-07-06

- **OpenArm 现场数据集 v1 本地冻结**: `/storage1t/ipc` 中 7 批 raw HDF5 已合并转换为
  `/storage1t/datasets/openarm_site_align_v1`，共 151 条、394900 帧、141/10 split；基础校验和 merge smoke
  通过，远端 SSH 恢复后同步到训练服务器并生成 norm stats。

## 2026-07-03

- **OpenArm 现场 HDF5 转 LeRobot v2.1 清洗入口**: 新增现场 raw HDF5 到 LeRobot v2.1 的转换脚本，
  默认统一 prompt、重编号、保留三路 mp4 且不重编码，并将 raw action 中缺失的左右夹爪维度按同帧
  state 夹爪值补齐；现场 v1 计划收敛为 100 条、90/10 split。

## 2026-07-02

- **OpenArm KAI0 适配计划重排**: Stage Advantage v1 以 `10000` checkpoint 作为当前最优结果进入可用状态；
  现场对齐数据冻结名收敛为 `openarm_site_align_v1`，150 条时先按 130/20 split 启动，扩到 200 条时改为
  180/20；下一轮训练主线调整为 `site_v1_ft_probe` -> `hq_tda_site_v1` -> `AWBC_v1`。
- **OpenArm site/TDA/AWBC 训练入口**: 新增 site-only probe、HQ/TDA/site 合并训练和 AWBC 训练配置；
  新增 LeRobot v2.1 多源合并脚本以及 Stage Advantage 到 AWBC 数据集构建脚本，用于在现场数据录制期间提前打通训练准备链路。
- **OpenArm TDA 源映射 AWBC**: TDA 增强脚本开始在 `episodes.jsonl` 写入源 episode 映射元数据；
  新增旧增强数据 metadata-only 修复脚本，以及从原始 HQ 预测 Stage Advantage 后映射到 TDA 增强 episode 的 AWBC 构建脚本，避免直接对镜像/抽帧视频重新打分造成越域误判。

## 2026-07-01

- **PyTorch 训练显存/速度调参开关**: `TrainConfig` 新增 `pytorch_gradient_checkpointing`，允许在 Stage
  Advantage 等 PyTorch 训练中显式关闭梯度检查点，用于在 A800 80GB 上用更多显存换取更高吞吐。
- **OpenArm KAI0 现场数据集命名澄清**: 明确 `openarms_folding_v001/v002` 不是现场数据集 v1，避免将既有
  OpenArms 折叠数据误作为 `site_v1_ft` 或 `hq_tda_site_v1` 的现场输入。

## 2026-06-30

- **OpenArm HQ + TDA 首轮真机反馈**: IPC 端 OpenPI HQ 推理已用客户端 `tda_smooth` 跑通，起身到桌面阶段表现良好；当前失败主因是主摄像头与 HQ 数据集分布差异较大导致夹爪抓取不准，后续优先做相机视角/数据分布对齐。
- **OpenArm HQ TDA 数据增强**: 新增 `scripts/augment_openarm_hq_tda.py`，面向服务器上已转换好的
  LeRobot v2.1 HQ 数据集生成 `openarm_hq_tda_aug_v1`，显式适配 OpenArm 16D 左右臂互换，并优先使用
  ffmpeg CUDA/NVENC 处理 time-scaling 与镜像视频；当节点 NVENC 不可用时会提前报错，或在显式允许时改用
  CPU encode + GPU decode。
- **`pi05_openarms_dual_hq_tda_aug` 训练配置**: 新增 TDA 增强数据集训练配置，目标数据集为
  `/share/home/linyongjia/datasets/openarm_hq_tda_aug_v1`，默认从 HQ `99999/params` checkpoint warm start。

## 2026-06-29

- **训练服务器链路收敛**: 当前默认入口统一为 mu01 (`172.31.11.100:12222`) 二跳到 gpu12/gpu14；
  gpu08 (`172.31.11.108`) 标记为 Slurm 受限历史节点，不再作为默认训练入口。
- **本地/训练服务器代码同步**: `conda-pi` 本地、origin 与训练服务器共享目录已按最新提交同步；远端未提交的
  多节点训练改动已先保存为 stash 后 fast-forward，避免覆盖服务器现场改动。

## 2026-06-27

- **记忆系统接管**: 将仓库记忆统一为 `AGENTS.md` + `docs/cache/` 的单一 Context OS 入口，补齐
  `docs/decisions/README.md` 与 `docs/reference/00_reference_index.md`，清理 `.agents`/`.codex`
  空占位，并将当前远端训练默认链路收敛到 `/share/home/linyongjia/data` + `local/<alias>`。
- **训练节点记忆修正**: 补充 gpu12 (`172.31.11.112`) 为可访问训练节点；当时默认集群记录为 gpu08 +
  gpu12，两节点共 4 张 A800。
- **gpu14 节点验证**: 确认 gpu14 (`172.31.11.114`) 需经 mu01 (`172.31.11.100`) 二跳进入；
  `nvidia-smi` 和 JAX 均识别 2 张 A800，可作为 OpenPI/JAX 训练候选节点。
- **JAX 4 卡多节点适配**: `scripts/train.py` 支持早期 JAX distributed 初始化、coordinator bind address、
  非主进程跳过 wandb/metrics 写入；Torch/LeRobot JAX loader 按 `jax.process_count()` 分片并使用
  `DistributedSampler`；checkpoint resume 显式按当前 sharding restore，避免 2 卡旧 checkpoint 在 4 卡拓扑
  上读取旧 sharding 文件时报 `no addressable shards`。

## 2026-06-25

- **`pi05_openarms_dual_hq` 训练配置**: 基于 robot-folding 博客消融实验结论，新建 HQ 数据集 (1200 集) 的 π0.5 微调配
  置。关键参数：`base_image_key="observation.images.base"` (HQ 数据集用 `base` 非 `top_rgb`)，
  `action_style="relative"` (UMI-style，博客验证 +15%)，`train_episodes=list(range(1000))`
  (openpi 不自动读取 LeRobot splits)，`num_train_steps=100_000` (博客 recipe)。
- **`evaluate_checkpoint.py` 修复**: 不再硬编码 `top_rgb`，自动检测 `base`/`top_rgb` 相机 key；
  修复 `delta_timestamps=None` 导致 action 无法比对 ground truth 的 bug；
  新增 `task_index→prompt` 映射支持。
- **全流程审查**: 发现两个关键路径问题：(1) norm_stats 计算写入数据集目录但训练从
  `assets/{config}/{asset_id}` 加载，需手动 `ln -sf` 桥接；(2) openpi DataLoader 不自动使用
  LeRobot info.json 的 splits，需 `DataConfig.train_episodes` 显式指定。
- **HQ 数据集上传**: 85GB `high_quality_folding` 从本地 `/storage1t` 上传至服务器
  `/share/home/linyongjia/datasets/`。
- **记忆系统更新**: 新增 `hq-training-workflow` 记忆和 `deployment.md` HQ 训练章节。

## 2026-06-24

- **Context OS 记忆系统初始化**: 建立四层 Context OS 记忆架构 (CLAUDE.md → cache/ → memory/ → CHANGELOG)，将服务器地址、远端路径、GCS 模型位置等核心事实固化到跨会话记忆中。
- **训练服务器 IP 更新（历史）**: 当时从 `172.31.11.122` 改为 `172.31.11.108` (gpu08)；当前默认以最新条目为准。

## 2026-06 (近期)

- **离线 WandB 支持**: 训练强制使用离线 W&B 模式，同时生成本地 metrics.jsonl/metrics.csv/训练曲线图。
- **Piper 双臂 Conda 训练**: `conda-pi` 分支建立非容器 conda 环境的离线训练路径，支持 `pi05_piper_dual` 配置。
