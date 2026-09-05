# 07 · 变更历史

只记录已经发生的变更、原因、结果和必要的证据索引。这里出现的旧路径、step、指标或命令都是历史快照，不能直接当作当前默认值；当前操作以 `docs/00`–`docs/06` 和源码为准。

当前架构 → `docs/01_system_architecture.md`；研究计划 → `docs/06_openarm_research_plan.md`；交接入口 → `docs/00_handoff_index.md`。

---

## 2026-09-05

- **本地 Pi0.5 资产补齐**：本地 `condapi-yam`（Python 3.12.12）已安装并通过 `pip check`；Pi0.5 基础
  checkpoint 已确认缓存完整，PaliGemma tokenizer 已下载并完成实际编码 smoke。当前 `pi05_yam_lora` 不需要
  额外的 Pi0、Pi0-Fast 或 FAST tokenizer 权重。
- **本地验证结果**：YAM policy 定向测试 4 项通过；排除大模型实例化后的轻量测试为 `53 passed, 2 deselected`。
  本地 CPU 执行完整模型测试曾因内存限制以 exit 137 结束，模型创建、GPU smoke 和正式训练留待服务器 GPU 可用后复核。

## 2026-09-04

- **项目默认路线切换为 YAM**：当前仓库从 OpenArm 默认改为 YAM 双臂（与 YAM-ABC 同硬件配置）训练适配，新增独立 `LeRobotYamDataConfig`、`YamInputs/Outputs` 和 `pi0/pi05_yam(_lora)` 配置；首选 `pi05_yam_lora`，模型内部 32D/50 步，YAM 输出 14D。独立 YAM-ABC-Reproduce 代码未同步进本仓库。
- **YAM 数据加载兼容**：按 YAM LeRobot 合同接入 `observation.state`、`action` 和三路 RGB 键，新增 LeRobot v3 import/task metadata 兼容；依赖固定到 `lerobot==0.5.1`、`torchcodec`，项目环境改为 Python 3.12 以满足该版本约束。
- **环境与同步规则更新**：本地安装 Miniconda 和 `condapi-yam` 验证环境；服务器项目环境规划为 `/home/wuyan/.conda/envs/condapi-yam`。每次中文 commit 后同时推送 Gitea `origin` 和 GitHub `github` 互为备份。
- **污染审计**：发现并按当前项目规则重写了外部 agent 对 `README.md` 的 YAM 改动，恢复被删除的 `docs/00_handoff_index.md`；未发现其他不属于本轮 YAM 适配的未跟踪文件。

- **服务器切换接管**：根据琶洲模方智算平台手册和只读 SSH/Slurm 核验，将当前默认服务器入口更新为 `wuyan@10.18.31.234:22`（登录主机 `rocky-login.hlink.local`），工作台入口为 `http://10.18.31.233:3080/`；用户密码未写入仓库。
- **新平台目录核实**：确认家目录 `/home/wuyan`、项目工作区 `/home/wuyan/lyj/YAM`、数据目录 `YAM_data` 和代码目录 `YAM_code`（当时为空）。
- **下载任务保护**：接管时 Slurm `1962/abc-download` 在 `gpu001` 运行，`ABC-130k-two-tasks` 数据目录约 23 GB，日志显示视频片段处理到 `3200/19917`、失败数为 0；本次未停止、重启、删除或修改远端任务和数据。

---

## 2026-08-28

- **交接文档第三轮收敛**：README 改为 OpenArm 主线最短入口；补齐训练、服务器、服务参数和初始/复位位姿归属（含 `/home/lyj/openarm_ros2_docker` 回零入口）；离线评估固定使用 HQ `999:1199` holdout，并明确 `rollout_drift` 预留字段不作 gate；`train_test.py` 调用改为可执行的 pytest 入口。
- **清理无职责文档**：删除空的 `docs/decisions/README.md`；编号化主线仍由 `docs/00`–`docs/07` 按唯一职责维护，Piper 只保留 reference 指引。
- **验证结果**：Markdown 链接/锚点 28 项、Bash 文档块 33 项和 Python compileall 通过；外部 OpenArm ROS 脚本语法通过；smoke/launcher 目标测试 7 项通过。当前本机 `pi-conda` 未安装 `lerobot`，数据 audit、训练和评估命令需在完整离线包环境复核。

---

## 2026-08-25

- **服务器与数据集交接手册加深**：将远端拓扑、跳板登录、GPU/tmux 审计、K-Data 1719 集硬门禁、norm stats、真实 loader、四卡 smoke/80k、总控和 checkpoint 交接集中到 `docs/02`；将服务健康检查、WebSocket 合同、RTC 回退、HIL raw/clean 和 rollout 停止条件集中到 `docs/05`。本轮未启动/停止/删除服务器任务；SSH 只读探测被远端在密钥交换阶段关闭，因此实时占用和服务状态仍须按 `docs/02` 第 2、3 节现场复核。
- **重复入口清理**：删除无职责的 `docs/README.md` 和重复 RTC 参考页，`docs/reference/` 只保留 Piper legacy 指引；当前 RTC、OpenArm 合同和服务器命令分别以编号化主线文档和源码为准。
- **交接文档三轮重构**：将 README、Context OS、安装/训练/数据/推理文档改为 OpenArm 主线的 00–07 编号体系；Piper 长篇训练、微调、推理和专用 RTC 说明删除，只保留一个 legacy 指引；旧上游说明删除，避免与当前 OpenArm 合同冲突。

---

## 2026-07-27

- **K-Policy 真机覆盖与阶段审计**: 20k 真机表现未优于79999，gpu25恢复为79999运行候选并保留离线20k
  正式选择历史；全量标签审计确认折叠阶段帧数和positive数量均不少于展开阶段，当前阶段切换失败的关键
  边界是 `stage_id_awbc` 不进入策略条件且成功示范缺少错误甩平/恢复状态，下一批改为30条定向完整HIL。
- **后续路线收敛**: 当前采集器固定为K-Policy79999，首批定向HIL按错误对角线、重复甩平和已展开不折叠
  各10条采集；主力路线使用同一批数据训练Evo value/ACP并从79999初始化组合策略，Site-5K只保留为
  Evo受控对照，Site-heavy采样和阶段条件提示词作为隔离消融而非官方复现。

## 2026-07-22

- **K-Policy HQ 评估坏视频恢复**: 80k 训练完成后的 HQ sweep 抽到缺少有效 MP4 `moov` 元数据的
  多个左腕视频（包括 episode 1170/1177/1186）而退出；评估器现有限重试视频探测，对持续不可解码的已抽中 episode 做
  确定性替换并原子记录选择与拒绝原因，同时保持全部有效的原抽样和既有 Site 评估缓存不变，无需重跑训练。

## 2026-07-20

- **KAI0 完成审计闸门**: 总控新增独立 completion audit，只有 HQ999、Site150、K-Data、最终79999、
  16-checkpoint sweep、选中权重、gpu25真实推理 smoke 与报告一致性全部通过才允许标记 complete。
- **K-Policy 关键帧评估修正**: checkpoint sweep 的关键帧从关节绝对角度范数改为相邻帧动作变化与夹爪
  状态变化，避免把姿态幅度大的静止帧误当作抓取、抬升或夹爪切换关键帧而影响最终选模；采样指纹记录
  选择器版本，旧算法报告不会被断点续跑误复用。
- **K-Policy 部署验收加固**: gpu25 部署不再仅以端口监听为成功；新增 OpenArm WebSocket 真实推理 smoke，
  强制校验有限的 `(50,16)` 动作以及50步/16D/角度制/夹爪 `0/-66` 元数据，并把选中 checkpoint、
  positive prompt、时延和动作摘要写入部署报告。旧 checkpoint 或缺少 smoke 的部署记录不会被误复用。

## 2026-07-13

- **K-Data HQ 范围校验修复**: Stage 评分派生集会统一训练提示词，无法保留原始 HQ `360:536` 的 layout
  任务证据；正式构建现从原始 HQ episode metadata 校验固定任务范围，并同步核对评分与源数据的 episode ID
  和逐集长度，避免误拒绝正确评分或静默混入错误数据版本。
- **Site-Stage 适配门禁修正**: 适配后 Site 分阶段使用人工边界、正式标签使用 `absolute_advantage`，因此硬门禁
  改为绝对进度 MSE/MAE/correlation/R²；局部相对方向和模型 crossing 保留为诊断，避免用未进入正式
  K-Data 的指标阻塞已经通过双域 paired-frame 验证的 checkpoint。
- **K-Data 混合视频预检修复**: 四卡 smoke 随机命中 Site 视频后暴露毫秒级时间戳偏差；K-Policy 现使用
  与现场数据一致的 LeRobot 容差，正式 loader 审计扩展为解码 HQ/Site/TDA 各自首、中、尾样本，避免
  只验证数据集开头的 HQ 视频而漏过混合来源问题。

## 2026-07-16

- **K-Policy TDA 尾帧解码修复**: 四卡主训练五次在固定 step 984 后命中同一 TDA 样本；其 MP4 容器头比
  实际内容多一帧，TorchCodec 严格索引越界。全量 PyAV 虽正确但实测约 25 秒/step，不适合 80k；正式
  loader 改为 TorchCodec 快路径，仅在明确的 end-of-stream 尾帧异常时对当前样本回退 PyAV。审计强制
  解码全部 300 集 TDA 的尾帧。
- **K-Policy 六卡训练适配**: 当前空闲同构节点为 gpu12/gpu14/gpu28，启动器支持可配置双卡节点集合；
  默认 smoke/80k 为三节点六卡、global batch126，并将全部远端会话作为不可拆分的生命周期单元；监督器
  支持通过环境变量覆盖节点、批量和实验标签，节点临时被占用时可退回空闲双节点四卡/global batch128。
- **K-Policy loader 吞吐调优**: gpu28、本地 batch64 的 TorchCodec 定向回退基准显示 workers2 六批
  平均 17.68s，workers4 十四批平均 14.38s，workers8 十四批平均 8.74s，workers16 首批超过 3 分钟；
  正式 80k 使用每进程 workers8，监督器可通过 `OPENPI_K_NUM_WORKERS` 覆盖。

## 2026-07-12

- **Site-Stage 自动衔接修复**: Site150 评分完成后，监督器因派生数据缺少 `episodes_stats.jsonl` 和现场
  视频毫秒级时间戳偏差未写迁移决策；构建器现生成逐集统计，旧产物可自动回填，曲线闸门失败时短路进入
  Site-Stage，Stage loader 显式使用现场容差。直接迁移确认不合格后已自动启动条件适配。
- **Site-Stage 显存配置修复**: 双卡 global batch64、关闭梯度检查点时峰值约 78.7 GiB/卡并 OOM；条件
  适配改为 global batch32 并使用新实验目录，首步双卡约 67.9 GiB/卡，保留后续自动评估和重试链路。

## 2026-07-11

- **远端 supervisor SSH 参数边界修复**: Site scorer 自动接棒暴露 OpenSSH 会把 trailing argv 重新拼成
  远端 shell 字符串，未转义的 `&&`/重定向导致 tmux 只收到 `cd`、实际命令在 home 下失败；Site
  supervisor、KAI0 总控和 4 卡 JAX launcher 现统一用 `shlex.join` 保留远端 argv 边界，并补齐回归测试。
- **Site watchdog 重试上限修复**: 远端启动抛异常时也计入连续重试，达到三次后显式进入
  `restart_exhausted`，不再每五分钟无限刷同一错误；任一完整 episode 产生进度后仍自动清零重试计数。

## 2026-07-10

- **Stage 单样本尾批修复**: 修正 PyTorch `resize_with_pad` 将输入为 `(1,H,W,C)` 的真实 batch 误当作
  临时扩维并 squeeze 的问题；Stage 评分在 episode 尾批只有一个 pair 时现保持 4D，避免 shard 中途
  因 `permute` 维数错误退出，同时补齐 channels-first/last 的批量与非批量形状测试。
- **Stage 视频瞬时 I/O 恢复**: HQ 六路评分遇到共享盘 MP4 的 OpenCV 临时打开超时后，视频读取器现对
  打开和单帧读取执行有限退避重试；持续失败仍明确报错并由 episode 级 watchdog 续跑，避免瞬时 NFS
  抖动每次都终止整个 shard。
- **JAX 长训采样与断点恢复修复**: Torch/LeRobot JAX loader 改为按 epoch 确定性重洗牌，多机分片保留
  rank 互斥并支持批次偏移；`train.py` 恢复权重后按 `train_state.step` 定位下一批，避免 80k 每轮重复
  固定顺序，以及从 5k checkpoint 恢复时重新读取 batch zero。
- **K-Policy 综合训练报告**: 新增自包含响应式 HTML，汇总训练曲线、16-checkpoint 双域指标、K-Data
  标签分布、质量闸门、选中权重和部署合同；桌面/移动 Playwright 渲染通过，最终由 gpu28:8769 服务。
- **16-checkpoint sweep 可恢复化**: HQ/Site sampled sweep 逐 checkpoint 原子写 v2 报告并严格续跑；缓存
  只有在权重、数据、采样配置和 positive prompt 全匹配时复用，选模端再次拒绝旧 schema/错误 prompt。
- **KAI0 正式 advantage 源对齐**: 对照官方主 README、AWBC README 和离散脚本后，正式 K-Data 改为
  每阶段按 `absolute_advantage` 取 top-30%；`relative_advantage` 继续落盘并做非塌缩诊断，但不再决定二值标签。
- **K-Data norm 原子落盘**: parquet norm stats 改为临时文件完成后原子替换，避免长时间全量统计在最终
  JSON 写入瞬间中断后留下截断文件，使无人值守总控能够安全重试。
- **JAX 异步 checkpoint 完整性闸门**: K-Policy smoke/full 与 sweep 只认含 Orbax 顶层及 params
  完成 metadata 的 checkpoint，忽略异步保存中的半成品数字目录；成对恢复会回到最近完整的 5k 权重。
- **K-Policy 选模与部署语义修复**: checkpoint sweep 明确按 80k 主循环实际
  `5000/10000/.../75000/79999` 目录选择全部16个候选；新增可选 `--force-prompt`，正式 K-Policy 服务强制 positive 条件，
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
  `docs/06_openarm_research_plan.md` 为当前 OpenArm RECAP/Evo-RL 复现执行计划；旧训练流水和事故细节不再放在热路径。
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
