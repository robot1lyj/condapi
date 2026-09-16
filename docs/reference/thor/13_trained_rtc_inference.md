# 13 · Pi0.5 training-time RTC：Thor 转换与推理候选

2026-09-16 状态：**代码候选、Pi 系列 RTC 候选镜像与 Thor 隔离容器 CPU 合约测试已完成；尚未取得该训练 run 的完整 RTC checkpoint，未做真实 JAX→Torch→TensorRT 精度/延迟验收，也未启动 RTC 服务。** 现行 `pi05-infer` 的 100000 W/80 服务不变。RTC 新服务预留独立 8001 端口，验收前不得以它替换 8000。

## 算法合同：没有兼容降级

依据 [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/html/2512.05964v1)，训练时把同一示范块的前 `d` 个动作设为干净动作，OpenPI flow time 为 `0`，只对后缀计损失。当前训练 `d∈[0,10]`、H50、32D、十步 Euler；模型没有因 RTC 增加权重。新采样器每一步都把已承诺前缀放回原动作值、前缀 token 的时间固定为 `0`，后缀使用当前去噪时间；只返回以新 observation 为条件的后缀预测。它**不是**旧 `sample_actions_rtc` 的推理时梯度 guidance，也没有默认改成普通同步推理。

输入前缀由控制侧给出，是已经决定执行、且从新块第 0 个目标 tick 开始的 `(d,14)` 绝对动作；`committed_start_tick == target_start_tick`。Thor 用**新 observation 的 14D state**把12个关节转换为 delta，两个夹爪仍 absolute，再用本 checkpoint norm 归一化、补齐 32D。返回经过现有逆变换的 `(50,14)` 绝对目标；前 `d` 个物理动作直接保持控制侧原值，不让浮点往返改变已承诺命令。时序所有权仍在3588，Thor不控制机械臂、不采集相机。这里的 tick 是控制侧目标动作 tick，不应把图像帧时间、请求到达时间或未知的第0步 observation 偏移假设为同一个量。

RTC WebSocket 用 `{"type":"infer","obs":{...},"rtc":{"delay_steps":d,"target_start_tick":N,"committed_start_tick":N,"committed_actions":[...]}}`；`obs` 含三路 RGB、14D state、prompt。`d=0` 时提交空列表。任何 `d>训练最大值`、尺寸/非有限、tick 错位均拒绝；不裁剪、不补齐、不静默普通推理。`rtc_mode="trained"` 的严格分支与旧 `off/only` 路线分离。

## 固定模型路径

1. 从完整、有限的原始 JAX/Orbax RTC checkpoint 及对应 `training_contract.json`、checkpoint 内 `assets/yam/norm_stats.json` 开始。`scripts/thor/prepare_rtc_checkpoint.py --checkpoint <JAX目录> --training-contract <训练合同> --output <新PyTorch目录>` 核对 `pi05/H50/32D/dmax` 与 norm SHA，复用现有审计转换器保存 FP32，不丢精度、不丢参数。结构未新增权重，因此无需另写 RTC 权重映射。转换成功只表示权重可加载，不表示 RTC 数值一致。
2. 制作真实 YAM `cases.json`：`source_kind=real_yam_recording`、`norm_stats_sha256`、`cases` 列表。每行含真实 observation `sample`、其 `provenance`、同 episode 连续执行的 `committed_actions` `.npy`、`committed_actions_sha256`、`source_episode`（provenance `source_files` 中的精确路径）、`delay_steps`、`target_start_tick`、`committed_start_tick`。至少覆盖 `d=0/1/dmax`、早/中/晚片段；从 episode 的动作数据提取而不是让模型伪造前缀。所有路径相对 cases 文件目录。确认样本的 observation 与前缀确实来自同一训练数据时间窗后才用于精度报告。
3. 原始 JAX 生成金标准：`scripts/thor/rtc_jax_reference.py --checkpoint <JAX目录> --training-contract <合同> --cases <cases.json> --output <新reference目录>`。固定同一噪声、原始FP32参数/计算、完整YAM逆变换。Thor GPU 上按推理/测试规则启用 MAXN，并在会话结束恢复日常功耗。
4. 转换后的 PyTorch：`scripts/thor/export_pi05_rtc_onnx.py --checkpoint <PyTorch目录> --cases <cases.json> --jax-reference <reference目录> --output <新export目录> --compute-dtype float32`。先验 Torch eager 与固定形状 RTC wrapper 同噪声**逐位一致**；对 JAX 记录归一化32D和物理14D误差，FP32 最大绝对差数值门槛各 `1e-4`（仅数值排错门槛，绝非任务效果门槛）。然后导出 ONNX，确认 `previous_actions` 与 `prefix_mask` 是运行时输入。BF16 是独立候选，必须重新走同样测试，不能拿 FP32 成绩代表 BF16。
5. 已通过 JAX 门槛的 ONNX 用既有 `scripts/thor/build_trt_engine.py` 构建无量化、strongly typed、TF32-off 引擎。RTC TensorRT 运行时新增两个固定尺寸输入 `[1,50,32]` 和 bool `[1,50]`，**内容动态**；固定形状允许每次请求更新 graph input buffer 后 CUDA Graph replay，不要求放弃 Graph。时间条件缓存实现了干净 `t=0` 与后缀十步两套投影，但**默认关闭**，待完整 checkpoint 对比逐位相等后才作为单独加速实验；不把它作为算法必需。
6. 已验证的引擎可由独立入口 `scripts/thor/serve_pi05_rtc_trt.py` 加载；必须提供真实 warmup observation/RTC payload。容器仍按 Pi 系列隔离，不新建模型系列容器。先做 Thor 本机/直连协议 smoke、不同 `d` 的前缀不变与后缀变化、非有限检查、Graph/普通执行一致、延迟和真实任务回放；现场闭环另由用户测试。仅完成 build 不等于能上线。

## 实现与已验证范围

JAX参考采样 `src/openpi/models/pi0.py::sample_actions_trained_rtc`；Torch eager `src/openpi/models_pytorch/pi0_pytorch.py::sample_actions_trained_rtc`；固定形状 RTC wrapper `scripts/thor/rtc_onnx_sampler.py`；绝对动作/归一化往返 `scripts/thor/rtc_action_space.py`；严格请求与现有 policy transform 复用 `scripts/thor/rtc_policy.py`；TensorRT适配及服务见同目录 `rtc_trt_policy.py`、`serve_pi05_rtc_trt.py`。现有100000 `serve_pi05_trt.py`、旧 W engine、旧 WebSocket `rtc_mode=off` 均未改为 RTC。

2026-09-16 在 Thor 原 Pi v6 镜像的**隔离 CPU 测试容器**覆盖 6 项：标量/逐 token 时间嵌入、adaRMS 逐 token 调制、clean prefix 与十步后缀、eager/wrapper 同值、条件缓存选择、YAM 绝对动作往返和请求 tick 校验；6/6 通过。容器故意不挂 GPU，因此 NVIDIA 启动横幅的 CUDA 初始化提示不是推理故障。它不覆盖真权重、ONNX、TRT、CUDA Graph 或机械臂。首个完整 RTC 保存点到来后再执行上述真实链路，不得用非 RTC 100000 冒充。

同日已在 Thor 用 `scripts/thor/Dockerfile.pi05-rtc` 基于原 Pi v6 镜像构建 `openpi-pi:thor-trained-rtc-candidate-20260916`，镜像 ID `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`。Dockerfile 只 overlay 新的 OpenPI JAX/Torch、配置、Gemma norm 和 WebSocket 源文件，不内置模型/凭据；构建时 RTC 方法 import 检查通过。在新镜像中再次隔离 CPU 测试 6/6、RTC 转换/参考/导出/服务模块 import 通过。构建上下文的具名源文件暂存于 Thor `/home/wuyan-lyj/thor/pi/probes/rtc-candidate-20260916/`；重建时用本仓库同名文件生成上下文，不把旧暂存内容当权威源码。旧 `pi05-infer` 检查仍为 running，镜像仍是 `openpi-pi:thor-pytorch-onnx-v6-20260907`、restart unless-stopped；没有更换、重启或修改原服务。
