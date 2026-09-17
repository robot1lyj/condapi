# 13 · Pi0.5 training-time RTC：Thor 转换与推理候选

2026-09-17 状态：**代码候选、Pi 系列 RTC 候选镜像与 Thor 隔离容器 CPU 合约测试已完成；服务器 20000 保存点已确认提交，正在向 Thor 续传，尚未完成真实 JAX→Torch→TensorRT 精度/延迟验收，也未启动 RTC 服务。** 现行 `pi05-infer` 的 100000 W/80 服务不变。RTC 新服务预留独立 8001 端口，验收前不得以它替换 8000。

## 算法合同：没有兼容降级

依据 [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/html/2512.05964v1)，训练时把同一示范块的前 `d` 个动作设为干净动作，OpenPI flow time 为 `0`，只对后缀计损失。当前训练 `d∈[0,10]`、H50、32D、十步 Euler；模型没有因 RTC 增加权重。新采样器每一步都把已承诺前缀放回原动作值、前缀 token 的时间固定为 `0`，后缀使用当前去噪时间；只返回以新 observation 为条件的后缀预测。它**不是**旧 `sample_actions_rtc` 的推理时梯度 guidance，也没有默认改成普通同步推理。

输入前缀由控制侧给出，是已经决定执行、且从新块第 0 个目标 **30 Hz policy tick** 开始的 `(d,14)` 绝对动作；`observation_policy_tick == target_start_tick == committed_start_tick`。这里的 policy tick 不是机械臂底层伺服 tick。Thor 用**新 observation 的 14D state**把12个关节转换为 delta，两个夹爪仍 absolute，再用本 checkpoint norm 归一化、补齐 32D。返回经过现有逆变换的 `(50,14)` 绝对目标；前 `d` 个物理动作直接保持控制侧原值，不让浮点往返改变已承诺命令。时序所有权仍在3588，Thor不控制机械臂、不采集相机。

**第0步的时间证据与未知项。** `src/openpi/training/data_loader.py::create_torch_dataset` 为 `action` 生成 `[0,1,...,49]/fps` 的取样偏移；本次 LeRobot 数据 fps=30。因此训练样本 `action[0]` 与 observation/state 取自同一**数据行时间索引** `t`，后续动作每33.33ms一格。[原始 RTC 论文](https://arxiv.org/html/2506.07339v2)定义 `A_t=[a_t,a_{t+1},...]`、`d=floor(推理耗时/控制周期)`；其脚注**明确假设**环境/底层控制器在消费 `a_{t-1}` 的同时提供 `o_t`，不考虑子 tick 延迟和同步问题。[训练时 RTC 论文](https://arxiv.org/html/2512.05964v1)沿用这个 controller-timestep 索引并以已承诺的 `A_{t:t+d}` 为前缀。因此本项目可确定的是**模型/数据索引** `action[0]↔observation tick t`，而不是相机曝光、编码/网络、3588控制队列相对于该 tick 的固定毫秒偏移。该物理映射必须由3588测量/定义，不能把相机采样时间或服务到达时间直接当目标 tick，也不能把数据行的0偏移宣称为真实设备0ms延迟。

客户端例子：若同步观测被归属 policy tick `k=100`，请求的第0步目标就是 tick100；推理期间旧队列已承诺 tick100–103，令 `d=4` 并提交这4个绝对动作，回包只从 `actions[4]` 的 tick104开始接管。结果早到就等到104；晚于104或实际已执行动作与提交前缀不同，就丢弃该结果并重新请求，不能事后修改 `d` 来套用旧结果。`d` 必须不超过该 checkpoint 训练上限10。若3588底层循环不是30Hz，先定义底层时间与30Hz policy tick 的映射，再构造 RTC 请求。

RTC WebSocket 用 `{"type":"infer","obs":{...},"rtc":{"delay_steps":d,"observation_policy_tick":N,"target_start_tick":N,"committed_start_tick":N,"committed_actions":[...]}}`；`obs` 含三路 RGB、14D state、prompt。`d=0` 时提交空列表。任何 `d>训练最大值`、尺寸/非有限、tick 错位均拒绝；不裁剪、不补齐、不静默普通推理。`rtc_mode="trained"` 的严格分支与旧 `off/only` 路线分离。

## 固定模型路径

1. 从完整、有限的原始 JAX/Orbax RTC checkpoint 及对应 `training_contract.json`、checkpoint 内 `assets/yam/norm_stats.json` 开始。`scripts/thor/prepare_rtc_checkpoint.py --checkpoint <JAX目录> --training-contract <训练合同> --output <新PyTorch目录>` 核对 `pi05/H50/32D/dmax` 与 norm 身份，复用现有审计转换器保存 FP32，不丢精度、不丢参数。容器运行时把审计转换器与本脚本同目录挂载，或显式传 `--converter`。结构未新增权重，因此无需另写 RTC 权重映射。转换成功只表示权重可加载，不表示 RTC 数值一致。
2. 制作真实 YAM `cases.json`：`source_kind=real_yam_recording`、`norm_stats_sha256`、`cases` 列表。每行含真实 observation `sample`、其 `provenance`、同 episode 连续执行的 `committed_actions` `.npy`、`committed_actions_sha256`、`source_episode`（provenance `source_files` 中的精确路径）、`delay_steps`、`observation_policy_tick`、`target_start_tick`、`committed_start_tick`。至少覆盖 `d=0/1/dmax`、早/中/晚片段；从 episode 的动作数据提取而不是让模型伪造前缀。所有路径相对 cases 文件目录。`scripts/thor/prepare_rtc_cases.py` 可从真实 Parquet 与已核验 RGB 回放生成，逐行核对 state 和30Hz时间索引；这仍是离线数据 tick，不证明现场物理同步。
3. 原始 JAX 生成金标准：`scripts/thor/rtc_jax_reference.py --checkpoint <JAX目录> --training-contract <合同> --cases <cases.json> --output <新reference目录>`。固定同一噪声、原始FP32参数/计算、完整YAM逆变换。Thor GPU 上按推理/测试规则启用 MAXN，并在会话结束恢复日常功耗。
4. 转换后的 PyTorch：`scripts/thor/export_pi05_rtc_onnx.py --checkpoint <PyTorch目录> --cases <cases.json> --jax-reference <reference目录> --output <新export目录> --compute-dtype float32`。先验 Torch eager 与固定形状 RTC wrapper 同噪声**逐位一致**；对 JAX 记录归一化32D和物理14D误差，FP32 最大绝对差数值门槛各 `1e-4`（仅数值排错门槛，绝非任务效果门槛）。然后导出 ONNX，确认 `previous_actions` 与 `prefix_mask` 是运行时输入。BF16 是独立候选，必须重新走同样测试，不能拿 FP32 成绩代表 BF16。
5. 已通过 JAX 门槛的 ONNX 用既有 `scripts/thor/build_trt_engine.py` 构建无量化、strongly typed、TF32-off 引擎。RTC TensorRT 运行时新增两个固定尺寸输入 `[1,50,32]` 和 bool `[1,50]`，**内容动态**；固定形状允许每次请求更新 graph input buffer 后 CUDA Graph replay，不要求放弃 Graph。时间条件缓存实现了干净 `t=0` 与后缀十步两套投影，但**默认关闭**，待完整 checkpoint 对比逐位相等后才作为单独加速实验；不把它作为算法必需。
6. 已验证的引擎可由独立入口 `scripts/thor/serve_pi05_rtc_trt.py` 加载；必须提供真实 warmup observation/RTC payload。容器仍按 Pi 系列隔离，不新建模型系列容器。先做 Thor 本机/直连协议 smoke、不同 `d` 的前缀不变与后缀变化、非有限检查、Graph/普通执行一致、延迟和真实任务回放；现场闭环另由用户测试。仅完成 build 不等于能上线。

## 实现与已验证范围

JAX参考采样 `src/openpi/models/pi0.py::sample_actions_trained_rtc`；Torch eager `src/openpi/models_pytorch/pi0_pytorch.py::sample_actions_trained_rtc`；固定形状 RTC wrapper `scripts/thor/rtc_onnx_sampler.py`；绝对动作/归一化往返 `scripts/thor/rtc_action_space.py`；严格请求与现有 policy transform 复用 `scripts/thor/rtc_policy.py`；TensorRT适配及服务见同目录 `rtc_trt_policy.py`、`serve_pi05_rtc_trt.py`。现有100000 `serve_pi05_trt.py`、旧 W engine、旧 WebSocket `rtc_mode=off` 均未改为 RTC。

2026-09-16 在 Thor 原 Pi v6 镜像的**隔离 CPU 测试容器**覆盖 6 项：标量/逐 token 时间嵌入、adaRMS 逐 token 调制、clean prefix 与十步后缀、eager/wrapper 同值、条件缓存选择、YAM 绝对动作往返和请求 tick 校验；6/6 通过。2026-09-17 在 RTC 候选镜像、正确的已安装 OpenPI 导入路径下复测，并加入 norm 末尾换行严格身份检查，7/7 通过。容器故意不挂 GPU，因此 NVIDIA 启动横幅的 CUDA 初始化提示不是推理故障。它不覆盖真权重、ONNX、TRT、CUDA Graph 或机械臂。首个完整 RTC 保存点到来后再执行上述真实链路，不得用非 RTC 100000 冒充。

同日已在 Thor 用 `scripts/thor/Dockerfile.pi05-rtc` 基于原 Pi v6 镜像构建 `openpi-pi:thor-trained-rtc-candidate-20260916`，镜像 ID `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`。Dockerfile 只 overlay 新的 OpenPI JAX/Torch、配置、Gemma norm 和 WebSocket 源文件，不内置模型/凭据；构建时 RTC 方法 import 检查通过。在新镜像中再次隔离 CPU 测试 6/6、RTC 转换/参考/导出/服务模块 import 通过。构建上下文的具名源文件暂存于 Thor `/home/wuyan-lyj/thor/pi/probes/rtc-candidate-20260916/`；重建时用本仓库同名文件生成上下文，不把旧暂存内容当权威源码。旧 `pi05-infer` 检查仍为 running，镜像仍是 `openpi-pi:thor-pytorch-onnx-v6-20260907`、restart unless-stopped；没有更换、重启或修改原服务。

## 2026-09-17 · 20000 保存点首次接入中

服务器原件：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_10h_20260916/20000`，`_CHECKPOINT_METADATA` 已含 `commit_timestamp_nsecs=1789600432477436453`，params 16个文件约12.44GB；本次只取 params/assets/完成元数据与训练合同，不取 train_state。Thor 独立目标：`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-10h-20260916/20000`，六个大 OCDBT 文件经本机 SSH agent 转发、Thor 直接从服务器 `rsync --partial --append-verify` 并行续传。源端仍在训练，选旧 20000 避开最新保存窗口；传输中不能恢复/转换。`thor-rtc-20000` 当前线程心跳负责续传、完成后核验与继续测试；状态不变静默，完成后停用。

发现一个可验证的 norm 序列化差异：训练源 `training_contract.json` 的 norm SHA256 是 `606d5c69e56aadb273ed3882ba9b62e11978a4222d8ff1e827e541bddd112893`；保存点 `assets/yam/norm_stats.json` 是 `b38ac082a729a4825c1a946c965183125382f10ccb5a75da89c2427019d1fefe`。服务器 `cmp` 核实前3416字节逐字节相同，训练源只多一个末尾换行；不是 norm 数值改变。`rtc_norm_identity.py` 仅接受原字节完全一致或**恰好少这一个末尾换行**，分别记录两种哈希；不接受任意 JSON 语义近似。旧 100000 norm 不能用于本次 RTC 回放。

Thor 已用 `prepare_rtc_cases.py` 生成新的真实离线回放 `/home/wuyan-lyj/thor/pi/test-data/pi05-rtc-20000-replay-20260917-r2`：源 Parquet SHA与原回放 provenance 一致，逐行 state 完全相同，9个三相机观测覆盖 episode95/96/97 的早中晚，`d=0/1/10` 各3个；prefix直接取同 episode 同行起的 `action`，不是模型伪造。`cases.json` 显式记录 `observation_policy_tick == target_start_tick`、数据时间戳和 norm 双哈希。它可供精度/延迟离线测试，不是3588物理时序已验收。
