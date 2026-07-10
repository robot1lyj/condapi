# openpi Change Log

Record **what changed + why**. Not config values, verification steps, or impact scope —
those live in config files, git log, and CI respectively.

Design decisions → `docs/decisions/`. Architecture → `docs/ARCHITECTURE.md`.

---

## 2026-07-10

- **K-Policy 选模与部署语义修复**: checkpoint sweep 按 OpenPI 实际 `4999/9999/.../79999` 目录选择全部
  16 个候选，不再误筛到只剩末步；新增可选 `--force-prompt`，正式 K-Policy 服务强制 positive 条件，
  避免客户端普通 task 覆盖 AWBC 推理语义，默认和 RTC 旧路径保持不变。
- **Stage 六卡评分预处理加速**: 三路当前/历史图像由逐帧 GPU resize 改为整批 resize，数值等价测试
  逐元素一致；HQ 单分片实测由约 8.5 降至 5.4 秒/batch，六路按 episode 原子断点恢复继续评分。
- **K-Data loader 预检修复**: 新增正式 1719 集结构审计和 positive/negative 真实 OpenPI loader smoke；预检
  发现构建器遗漏 LeRobot v2.1 必需的 `episodes_stats.jsonl`，现继承源统计并重算所有改写索引、二值标签
  和 Stage 分数字段，避免 norm 完成后才在四卡训练初始化阶段失败。
- **Site150 Stage 自动链路准备完成**: 150 条 success（排除 episode 95）按帧数均衡成 6 个 25 集分片；
  `kai0_site_score_monitor` 会在对应 HQ GPU 槽位释放后自动评分、恢复、审计并生成 Site 报告。另生成独立
  `openarm_site_stage_v1`（140 train + 10 val），只作迁移评估和条件 Site-Stage 监督，不作为最终分数。
- **正式二值 K-Data 构建器**: 新增 HQ999 + Site140×3 + 小预算 TDA300 的统一 AWBC 构建器；使用 KAI0
  逗号提示词和每阶段 top-30% 二值标签，并在落盘前检查完整 episode 集合、全局比例及各来源比例。
- **KAI0 无人值守训练总控**: 新增 Site 迁移失败后的条件 Stage 重训/双域 checkpoint 筛选、K-Data/norm、
  4 卡 JAX smoke/80k 自动恢复、HQ+Site checkpoint sweep 和 gpu25 部署状态机；AWBC sweep 支持强制
  positive prompt，避免用普通任务提示评估条件策略。
- **远端代码同步完整性修复**: 修复共享仓库只同步 `config.py`、遗漏新 `openarm_policy.py` 导致恢复任务导入
  失败的问题；远端现按完整 tracked file 集合同步，并明确共享 conda 导入校验必须在计算节点执行。
- **KAI0 混合任务阶段分桶修正**: 中途审计确认 HQ `536:999` 首帧已平铺、只包含 folding，低于0.5的
  episode-relative value 不是 scorer 崩坏；K-Data 改为 HQ `0:536` 模型 crossing、`536:999` 固定
  folding、Site 人工边界、TDA 源映射生成 `stage_id_awbc`，advantage 仍全部来自模型预测。
- **HQ999 Stage 正式质量闸门**: 新增完整 ID/有限值/分数范围、完整任务 crossing、folding-only 增量和
  relative advantage 非塌缩审计；HQ 动态报告对 folding-only 显示0.5阶段偏移并同时保留 raw 进度。
- **HQ 阶段范围合同与 Site 平衡审计**: K-Data 构建时强制验证 HQ `360:536` 连续 layout prompt 元数据；
  Site150 人工阶段帧比例确认为43.4%/56.6%，train/val 均无空阶段。
- **Stage 评分无人值守加固与报告重构**: score-only 数据集支持逐 episode 校验、原子 parquet 落盘和
  `--resume`；HQ watchdog 检测停止/日志停滞后最多自动恢复 3 次，并在 999 集完成后自动刷新最终报告。
  Stage 报告改为可复用渲染器，按相机原始宽高比在视频上叠加 progress/advantage、当前帧和正负状态。
- **Site-GT 删除并切换到模型评分**: 删除远端 `openarm_site_gt_v1` 与 8766 服务，保留 Site-A151；
  现场成功标注固定为 Site-A150，episode 95 固定为 Site-F1。先审计 HQ-Stage 对 Site 的直接迁移，只有
  迁移不合格才按 140 train + 10 val 适配 Site-Stage，最终 Site-Score 一律来自模型预测。
- **Site-GT/HQ-Stage 可视化对比**: 新增 HQ-Stage 已完成分片的 KAI0 Figure 4 风格动态报告；对比确认
  人工单边界 Site-GT 只能产生分段线性进度和近常数 advantage，因此暂停其直接进入 K-Data，等待确认
  Site 最终评分方案。
- **Site-GT 与动态审计报告**: Site-A151 已确定性生成独立 `openarm_site_gt_v1`，逐帧保留人工进度、阶段和
  50 帧 GT advantage，明确不调用 HQ-Stage；新增 KAI0 Figure 4 风格三路视频/曲线同步 HTML，并为六卡
  HQ-Score 增加只监控不自动重启的 watchdog。
- **KAI0 HQ/Site 评分边界收敛**: 现有 Stage v1 固定命名为 HQ-Stage，只自动评分 HQ；Site 151 条完整人工边界
  固定命名为 Site-A151，并直接生成逐帧 Site-GT progress/advantage，不再评估、微调或复用 HQ-Stage 处理 Site。
- **KAI0 Stage 评分拆分为 score-only 与全局二值化**: 分片任务只生成原始 relative/absolute advantage，
  避免各 GPU 独立分桶造成阈值不一致；Site 151 集单边界标注和 HQ train `0:999` 六分片评分已并行启动，
  正式 AWBC 标签必须等待分片合并和 Site scorer 审计。
- **OpenArm 三路线复现计划**: 当前计划拆分为 KAI0、Evo-RL 和 KAI0+Evo-RL 组合三条独立路线；
  Stage v1 经代码和数据审计确认对齐 KAI0 Task A 的 flattening/folding 双阶段核心实现，同时显式记录
  10k 训练规模、旧三档 AWBC、relative/absolute advantage 等尚未完成或存在论文/官方代码差异的部分；
  KAI0 第一版保留小预算 TDA，禁止直接混入全部 2298 条增强数据。

## 2026-07-09

- **Context OS 记忆瘦身**: 压缩 `docs/cache` 热记忆默认值和部署模式，重写
  `docs/openarm_recap_reproduction_plan.md` 为当前 OpenArm RECAP/Evo-RL 复现执行计划；旧训练流水和事故细节不再放在热路径。
- **OpenArm/Piper 数据链路拆分**: 新增 `OpenArmInputs/OpenArmOutputs` 与
  `LeRobotOpenArmDataConfig`，OpenArm 训练配置不再复用 Piper transform；OpenArm 路径固定校验
  HQ 16D state/action，并禁止使用旧 Piper 14D `swap_left_right` 逻辑。
- **OpenArm JAX ACP prompt 适配**: 新增 JAX 数据 transform，按 `complementary_info.acp_indicator`
  在 tokenizer 前注入 `Advantage: positive/negative` prompt；新增
  `pi05_openarms_dual_evo_acp_hil_v1_probe` 训练配置，HIL clean 数据集名统一为 `openarm_hil_evo_v1`。
- **OpenArm HIL raw 转 Evo-RL clean 数据入口**: 新增 HIL HDF5 -> LeRobot v2.1 转换脚本，并接入
  `scripts/convert_openarm_hq_dataset.py from-hil-hdf5`；训练端按 `session_state/selected_source/authority_source`
  过滤 policy/human/hold，丢弃 `intervention_hold`/`hold` 帧并重写视频，确保 Evo-RL 的
  `is_intervention=1` 只表示真实 human VR 控制。

## 2026-07-08

- **OpenArm site_deg 训练完成**: HQ `99999` -> site_deg 5k 已保存 checkpoint `4999`；
  原始 π0.5 base -> site_deg 10k 已保存 checkpoint `9999`，后续按 gpu25 smoke 后真机 A/B 验证。
- **OpenArm site_deg 数据审计**: 远端全量复核 151 episodes / 394900 frames，train/val 为
  `0:141` / `141:151`，state/action 16D，关节 degree-like，夹爪 HQ motor degrees `[-66, 0]`。
- **OpenArm HIL 主线收敛**: 后续优先复现 π*0.6 / RECAP 与 Evo-RL 的
  success/intervention/recovery -> value/advantage -> ACP 闭环；KAI0 保留两阶段 Stage/AWBC
  辅助，不把复杂 failure stage 作为 v1 主标签。
- **OpenArm HIL window BC 删除**: 删除独立 `recovery_v1_probe` 思路；HIL 接管数据保留完整
  episode，只通过 value/advantage/`acp_indicator` 回写后进入 ACP/AWBC 训练。
- **OpenArm Evo-RL/KAI0 并行规划收敛**: 计划调整为 Evo-RL value/ACP 与 KAI0 Stage/AWBC 双线并行；
  当前 site_deg 真机测试未完整成功但动作明显改善，作为 HIL/rollout collector。客户端同一份原始记录同时保留
  Evo-RL 与 KAI0 所需信号；KAI0 专属 `stage_progress_gt`、advantage 和 `task_index/tasks.jsonl`
  由后处理生成，不再引入自研 `failure_stage` 或三分类主标签。
- **OpenArm HIL 客户端格式确认**: 客户端 HIL 输出固定为 HDF5 episode + 三路 mp4 + `meta/info.json`
  + `meta/episodes.jsonl`；每帧记录 16D HQ 合同 state/action 和真实 human VR 语义的 `is_intervention`；
  hold 切换等待帧只作为 raw debug 可选保留，严格 Evo-RL 训练集应丢弃，episode 级接管起止段不是训练硬依赖。

## 2026-07-07

- **OpenArm site 单位错配修复**: 确认 HQ 1200 集为 degree-like，而旧 `openarm_site_align_v1` 为
  radian-like + normalized gripper；现场 OpenPI 数据合同收敛为 arm joints degrees + HQ-style gripper
  motor degrees，转换脚本默认输出 `openarm_site_align_v1_deg`，旧 site 5k/base10k 候选不再作为可用真机模型。
- **OpenArm HQ 数据合同入口**: 新增 `scripts/convert_openarm_hq_dataset.py` 作为现场数据清洗统一入口；
  默认对齐 HQ task `Fold the T-shirt properly`，并将 raw gripper normalized `0.0/0.84`
  标定到 HQ motor degrees `-66/0`，避免 task prompt 和夹爪语义再次分裂。
- **OpenArm site deg 数据重转**: 远端 `/share/home/linyongjia/datasets/openarm_site_align_v1_deg`
  已用 HQ contract 重转，151 episodes；`norm_stats.json` 已按 train `0:141` 生成并经 OpenPI config load 验证。
- **OpenArm site deg 5k 重训启动**: gpu12 tmux `openarm_site_deg_5k_20260707` 已启动
  HQ `99999` -> site deg 5k，step 0 loss `0.2434`，两张 A800 均已占用约 73.6GB。
- **OpenArm 旧单位 checkpoint 清理**: 删除旧 `openarm_site_align_v1` 单位合同下的 site 5k、site 1k、
  base 1000 三个 checkpoint 目录，共释放约 126G；后续只保留 `site_deg` 新合同候选用于推理/真机测试。
- **OpenArm site loader/续训诊断**: 新增 LeRobot loader 压测脚本，确认 `torchcodec+2 workers` 是当前
  site 数据读取吞吐最优组合；原始 π0.5 base -> site 10k 训练在单位审计前暂停，避免继续消耗 GPU。

## 2026-07-06

- **OpenArm 现场数据集 v1 本地冻结**: `/storage1t/ipc` 中 7 批 raw HDF5 已合并转换为
  `/storage1t/datasets/openarm_site_align_v1`，共 151 条、394900 帧、141/10 split；基础校验和 merge smoke
  通过，远端 SSH 恢复后同步到训练服务器并生成 norm stats。
- **LeRobot v2.1 现场数据训练兼容**: norm stats 计算改为纯 parquet/pandas 路径，转换脚本补写
  `episodes_stats.jsonl`，并新增 episode stats 回填和视频时间戳辅助脚本，避免远端 OpenPI/JAX import 或
  LeRobot 元数据缺失阻塞训练。
- **OpenArm site 4 卡训练修复**: 修复多 host checkpoint 目录竞态；DataConfig 支持显式传入 LeRobot
  `tolerance_s`/`video_backend`，site probe 配置使用 `0.05s` 容差以兼容 torchcodec PTS 偏差。

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
