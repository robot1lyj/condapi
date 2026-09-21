# 13 · Pi0.5 training-time RTC：Thor 转换与推理候选

**2026-09-21最新取消指令**：用户取消20h70000及此前未完成的20h80000下载，已停止70000 rsync及`thor-rtc-20h70000-prepare.service`（inactive），永久删除Thor `checkpoints/lego-pi05-rtc-base-20h-20260918/70000`和`80000`两个未完成目录，约释放5.3GiB。20h目录现仅保留90000原始与FP32转换件；源服务器检查点、10h版本、脚本及历史报告未动。90000在线服务健康检查OK。**下文70000等待下载/自动转换的计划已取消，不再恢复执行**；需要时重新下载，已删片段不在回收站。

## 2026-09-21 · 20h90000在线，70000仅下载/CPU转换

14:56 CST现场复核：20h90000部署流水线已退出0，`pi05-rtc-infer`运行，`thor-pi-maxn-20h90000.service` active/MAXN，固定8000健康检查OK。九例协议smoke通过，七步/quantile前缀，权重SHA `222649178ff6649edfe2edfe9f27afab87857f65437acca93f8e8d9445805655`；原始回执`probes/rtc-20h-90000-20260921-r1/20h90000-smoke.json`。九例对JAX最大关节差0.000682rad、稳态推理P50/P95=186.78/190.72ms，本机协议往返197.94/210.06ms，不含3588真实链路验收。

用户随后要求拉取同20h run的70000转换测试，但明确选择**先下载转换，暂不暂停90000**。当前仅运行`thor-rtc-20h70000-prepare.service`默认CPU准备分支：等待下载完成→源端SHA校验→有限性审计→FP32转换→按70000 norm重新生成真实回放，完成后退出，不建ONNX/TRT、不跑GPU精度或延迟、不切服务。`--gpu`/`--test-and-restore`虽已准备但**未调用，未获本轮执行许可**，必须等用户安排测试窗口。

70000源与目标分别为`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_20h_20260918/70000`和`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-20h-20260918/70000`；完整元数据SHA `6973b04963bafedb54f8700cc7d1c3183d4d6ee023d54cc592b2b5f1f3d32ae9`，norm与20h90000同为`d5493ce7b71790526312779d8fbccc374b9569f14d88722682b0e7c8bfcf1d52`。六路rsync直拉，只取推理产物。暂存脚本/源文件SHA清单位于`probes/rtc-20h-70000-20260921-r1`。此记录时下载/准备已启动，尚未确认完成，不能误记为70000已通过推理验收。

## 2026-09-21 · 切换20小时90000（下载/部署流水线进行中）

用户完成10h/80000测试，要求停止20h/80000下载，改取20h/90000并转换后运行推理。精确停止本任务20h/80000的rsync与父进程，已下载片段保留，不当完整检查点。当前10h/80000服务已停止，宿主120W；**不能因9月21日较早的恢复记录而认为当前在线**。

新源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_20h_20260918/90000`，完整commit元数据SHA `76475832d4d4834596d31ba3b26d19e2f17289d21d91415e57e1d23b7e396615`。Thor目标`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-20h-20260918/90000`。六路直拉params/assets/元数据/合同，不取train_state；下载后源端SHA逐项核验。

20h合同SHA `42c635bd92535797a00bdbc63ac8b7c09d8ee6754d6cd24e7e93800319192ac2`；checkpoint norm SHA `d5493ce7b71790526312779d8fbccc374b9569f14d88722682b0e7c8bfcf1d52`，合同记录的训练norm SHA `a20dd630b620bddf291db8cbe69ee2cdd69cae66779011a69670e426020cff6d`。身份关系须由既有严格norm检查确认，不能沿用10h的norm或回放metadata。H50/32D/dmax10不变。

一次性部署作业`thor-rtc-20h90000-deploy.service`已启动，等待下载完成的合同及源文件清单，再SHA校验→有限性/FP32转换→以原始真实suite+Parquet重新生成20h norm回放→MAXN七步JAX/ONNX/TF32引擎/九例验证→同8000候选服务smoke。全部通过后才替换同名`pi05-rtc-infer`并启动`thor-pi-maxn-20h90000.service`维持在线MAXN，失败则停止推进，不放宽门槛。无需再次请示上线，因为本轮已明确授权运行。脚本和清单在`probes/rtc-20h-90000-20260921-r1/`；恢复时先看journal及新回执，不重复下载/转换。当前只确认流水线已启动，**尚未确认20h90000转换、测试通过或服务上线**；10h80000权重保留回退，未操作3588。

**最新状态（2026-09-20 10:18 CST）**：80000已完成FP32保真转换、七步TF32 TRT构建、九例JAX对照和四例事故回放，本机九例WebSocket全部通过；`pi05-rtc-infer`已替换80000，按暂停要求保持**停止/120W/8000无监听**。稳态推理P50/P95=186.86/187.33ms，本机协议往返194.12/205.03ms；九例最大关节差0.001115rad。30000推理权重/引擎/回退容器已删除，原始JAX/报告保留；60000权重/引擎保留但旧服务容器已由80000替换。详细指纹、门槛、协议证据与边界见[80000报告](../../reports/thor/rtc-80000-20260920/README.md)。下文“进行中”及“60000在线”均为历史阶段。

## 2026-09-20 · 80000接入与旧推理模型清理（进行中）

用户要求拉取同run的80000替换60000，并删除30000推理模型；追加确认100000非RTC也可删。现场60000容器仍停止、Thor为120W，保持暂停状态，不将“替换”解释为重新开放推理。80000源为`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_10h_20260916/80000`，已确认完整commit元数据；完成元数据SHA为`10c89477271a1e8a30208a94857592c8efecf5be852aa05b7a2dd38640e97fd0`。Thor直接六路rsync下载params/assets和合同，不取train_state，目标为同RTC检查点父目录的`80000`。

已永久删除停止容器`pi05-rtc-30000-preserved-20260918`、`30000-pytorch-fp32-r1/model.safetensors`、`rtc-30000-onnx-fp32-cache80-7step-20260917-r1/sampler.onnx.data`和`rtc-30000-trt-tf32-cache80-7step-20260917-r1/sampler.engine`；约释放37GiB。原始30000 JAX及小报告保留，需要回退时必须重建。100000原始params/转换权重/引擎和容器已在9月18日删除，本次复核无大模型文件，仅约71MB图结构/配置/报告，继续保留追溯证据。60000权重和产物未删。

一次性作业`thor-rtc-80000-test-20260920.service`等待下载合同标记及`probes/rtc-80000-20260920-r1/source.sha256`，然后校验源端17个文件SHA、CPU有限性审计/FP32转换、MAXN下七步JAX→ONNX→TF32 TRT九例及事故四例回放，结束恢复120W；不会自动启服务。脚本暂存同目录`rtc80000-test-20260920.sh`。**此记录只证明下载/流水线已启动，不代表80000已转换、通过测试或已替换60000**。下一步检查作业日志和新回执，验收后配置同地址服务并本机smoke，结束继续暂停。

**最新在线状态（2026-09-18 11:14 CST）**：用户授权上线60000；`pi05-rtc-infer`已切换60000七步FP32权重＋TF32，固定`ws://192.168.250.1:8000`，quantile前缀与0.2rad/tick保护不变，MAXN会话`thor-pi-maxn-60000.service` active。真实协议九例通过，权重指纹匹配60000、输出有限50×14、前缀保持。本机往返P50/P95=197.85/208.18ms；未做3588端到端或真机任务验收。30000容器停止保留为`pi05-rtc-30000-preserved-20260918`，不是并行在线服务。回执见[online-smoke](../../reports/thor/rtc-60000-20260918/online-smoke.json)。下文“恢复30000/60000未上线”为先前离线阶段记录。

## 2026-09-18 · 60000 RTC离线验收完成

60000完整下载且源端/Thor逐文件SHA一致，原始参数非有限0，FP32转换811张量逐位映射一致。按七步、分位数前缀、FP32权重＋TF32/缓存80/CUDA Graph，在MAXN独占GPU完成JAX对照与TensorRT构建、九例d=0/1/10回放及四例历史事故d=9诊断。九例最大关节差0.001711rad、物理P99差0.000514；稳态推理P50/P95=190.73/193.40ms，总处理194.52/197.54ms，不含网络。事故四例最大新步进0.103139rad/tick；第四例对JAX关节P99为0.001745，记录为诊断局限而非宣称所有输入均满足同一门槛。完整证据、指纹与原始产物路径见[60000报告](../../reports/thor/rtc-60000-20260918/README.md)。测试结束恢复原30000服务/MAXN，60000未上线，未做3588/真机任务验收。

## 2026-09-18 · 用户授权清理历史大产物

为60000 RTC接入释放空间：删除Thor普通100000原始`params`及转换后的`model.safetensors`，普通模型历史ONNX外部权重/TRT引擎、RLinf转换权重和realtime-vla转换PKL；另删除30000未采用的5/6/8/10步、BF16及无缓存FP32/TF32实验的ONNX外部权重和引擎。原目录中的配置、norm、审计、诊断和小体积测试报告保留，但**这些历史目录不再具备直接执行条件**，需要重新获取权重或重建。

8个已停止的旧服务/转换/审计容器已移除；当前`pi05-rtc-infer`保持运行，`/healthz=OK`。保留30000当前七步ONNX/引擎、20000/30000 RTC原始与FP32权重、基础模型和分词器、RTC/JAX容器镜像及全部回放数据。60000下载目录未删除或覆盖。删除为永久释放空间，不是回收站；可重建产物需重新转换/构建，100000本地副本不能就地恢复。

60000已核对服务器完整commit元数据，正直接下载至同run的`60000`目录，仅params/assets/完成元数据和合同，不取19GB train_state。用户已授权下载/转换后暂停30000做独占GPU/MAXN测试，结束恢复30000；此记录不代表60000已转换、验收或上线。

2026-09-18 09:33 CST 最新状态：**修正分位数前缀的30000 RTC七步服务已在 `ws://192.168.250.1:8000` 运行，Thor处于MAXN；本机原9例两轮、事故4例WebSocket测试通过。**握手含`rtc_prefix_norm=quantile`与`0.2 rad/tick`关节跳变拒绝；旧验证回执不能启服务。9月17日旧 JAX↔TensorRT验收及WebSocket smoke基于错误前缀，均为历史，不得当作当前RTC条件正确性证据。新运行回执、延迟偶发异常及未做3588/真机验收边界见[事故报告](../../reports/thor/rtc-prefix-quantile-incident-20260917.md#2026-09-18-更新新版服务与-thor-本机协议测试)。旧100000与10步容器停止保留，**旧10步也有同一前缀错误，不是安全回退方案。**

## 算法合同：没有兼容降级

依据 [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/html/2512.05964v1)，训练时把同一示范块的前 `d` 个动作设为干净动作，OpenPI flow time 为 `0`，只对后缀计损失。当前训练 `d∈[0,10]`、H50、32D、十步 Euler；模型没有因 RTC 增加权重。新采样器每一步都把已承诺前缀放回原动作值、前缀 token 的时间固定为 `0`，后缀使用当前去噪时间；只返回以新 observation 为条件的后缀预测。它**不是**旧 `sample_actions_rtc` 的推理时梯度 guidance，也没有默认改成普通同步推理。

输入前缀由控制侧给出，是已经决定执行、且从新块第 0 个目标 **30 Hz policy tick** 开始的 `(d,14)` 绝对动作；`observation_policy_tick == target_start_tick == committed_start_tick`。这里的 policy tick 不是机械臂底层伺服 tick。Thor 用**新 observation 的 14D state**把12个关节转换为 delta，两个夹爪仍 absolute，再用本 checkpoint norm 归一化、补齐 32D。返回经过现有逆变换的 `(50,14)` 绝对目标；前 `d` 个物理动作直接保持控制侧原值，不让浮点往返改变已承诺命令。时序所有权仍在3588，Thor不控制机械臂、不采集相机。

**第0步的时间证据与未知项。** `src/openpi/training/data_loader.py::create_torch_dataset` 为 `action` 生成 `[0,1,...,49]/fps` 的取样偏移；本次 LeRobot 数据 fps=30。因此训练样本 `action[0]` 与 observation/state 取自同一**数据行时间索引** `t`，后续动作每33.33ms一格。[原始 RTC 论文](https://arxiv.org/html/2506.07339v2)定义 `A_t=[a_t,a_{t+1},...]`、`d=floor(推理耗时/控制周期)`；其脚注**明确假设**环境/底层控制器在消费 `a_{t-1}` 的同时提供 `o_t`，不考虑子 tick 延迟和同步问题。[训练时 RTC 论文](https://arxiv.org/html/2512.05964v1)沿用这个 controller-timestep 索引并以已承诺的 `A_{t:t+d}` 为前缀。因此本项目可确定的是**模型/数据索引** `action[0]↔observation tick t`，而不是相机曝光、编码/网络、3588控制队列相对于该 tick 的固定毫秒偏移。该物理映射必须由3588测量/定义，不能把相机采样时间或服务到达时间直接当目标 tick，也不能把数据行的0偏移宣称为真实设备0ms延迟。

客户端例子：若同步观测被归属 policy tick `k=100`，请求的第0步目标就是 tick100；推理期间旧队列已承诺 tick100–103，令 `d=4` 并提交这4个绝对动作，回包只从 `actions[4]` 的 tick104开始接管。结果早到就等到104；晚于104或实际已执行动作与提交前缀不同，就丢弃该结果并重新请求，不能事后修改 `d` 来套用旧结果。`d` 必须不超过该 checkpoint 训练上限10。若3588底层循环不是30Hz，先定义底层时间与30Hz policy tick 的映射，再构造 RTC 请求。

RTC WebSocket 用 `{"type":"infer","obs":{...},"rtc":{"delay_steps":d,"observation_policy_tick":N,"target_start_tick":N,"committed_start_tick":N,"committed_actions":[...]}}`；`obs` 含三路 RGB、14D state、prompt。`d=0` 时提交空列表。任何 `d>训练最大值`、尺寸/非有限、tick 错位均拒绝；不裁剪、不补齐、不静默普通推理。`rtc_mode="trained"` 的严格分支与旧 `off/only` 路线分离。

## 固定模型路径

**当前操作优先级（2026-09-18）**：以下步骤描述完整工具链，其中十步/TF32-off是最初参考配置。新RTC检查点优先复用已验证的**7步、FP32权重＋TF32、80-token/时间缓存/CUDA Graph**配比，并对新权重独立验收；原JAX/FP32诊断仍关闭TF32。JAX参考、导出、构建及服务的步数必须一致。前缀编码必须从训练配置读取`use_quantile_norm`，Pi0.5/YAM为分位数，两端显式传`use_quantiles=True`；数值对照之外还要独立核对训练transform及前后缀交界。服务提供新版`--validation`和`--jax-reference`，保留0.2rad/tick跳变拒绝；具体参数以当前脚本帮助为准。普通非RTC检查点走[12](12_checkpoint_handoff.md)。本轮用户已反馈修正后的真机效果良好，属于定性反馈，不是跨模型/跨检查点成功率验收。

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

## 2026-09-17 · 20000 保存点首次接入

服务器原件：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_10h_20260916/20000`，`_CHECKPOINT_METADATA` 含 `commit_timestamp_nsecs=1789600432477436453`，params 16个文件；本次只取 params/assets/完成元数据与训练合同，不取 train_state。Thor 独立目标：`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-10h-20260916/20000`。六个大 OCDBT 文件经本机 SSH agent 转发、Thor 直接从服务器 `rsync --partial --append-verify` 并行续传；传输期间将管理 Wi-Fi 从2.4 GHz切为5 GHz，完成后两端 params/assets 共17个文件、文件字节总和 `12440600720`、逐文件 SHA256 全部一致，`_CHECKPOINT_METADATA` SHA256 均为 `68188c321087001dec95ca5b4cd0ca10fe485407c912e75c130106c4e5d4faad`，训练合同 SHA256 均为 `ef773631bb5dac8d4de055dc1d7161cf57c691b68ae50ec983e3a6297cb5ed5f`。服务器与Thor `du -sb` 目录大小因文件系统目录项开销不同而略有差异，不是权重差异。源端仍在训练，选旧 20000 避开最新保存窗口；原始参数审计及FP32转换结果见本节末。用户暂缓RTC测试，`thor-rtc-20000` 心跳已暂停，不会自行启动参考推理、导出或测速。

发现一个可验证的 norm 序列化差异：训练源 `training_contract.json` 的 norm SHA256 是 `606d5c69e56aadb273ed3882ba9b62e11978a4222d8ff1e827e541bddd112893`；保存点 `assets/yam/norm_stats.json` 是 `b38ac082a729a4825c1a946c965183125382f10ccb5a75da89c2427019d1fefe`。服务器 `cmp` 核实前3416字节逐字节相同，训练源只多一个末尾换行；不是 norm 数值改变。`rtc_norm_identity.py` 仅接受原字节完全一致或**恰好少这一个末尾换行**，分别记录两种哈希；不接受任意 JSON 语义近似。旧 100000 norm 不能用于本次 RTC 回放。

Thor 已用 `prepare_rtc_cases.py` 生成新的真实离线回放 `/home/wuyan-lyj/thor/pi/test-data/pi05-rtc-20000-replay-20260917-r2`：源 Parquet SHA与原回放 provenance 一致，逐行 state 完全相同，9个三相机观测覆盖 episode95/96/97 的早中晚，`d=0/1/10` 各3个；prefix直接取同 episode 同行起的 `action`，不是模型伪造。`cases.json` 显式记录 `observation_policy_tick == target_start_tick`、数据时间戳和 norm 双哈希。它可供精度/延迟离线测试，不是3588物理时序已验收。

原始JAX参数已在Thor候选容器中用 `audit_checkpoint_finite.py`、原dtype完整恢复审计：全部参数非有限元素0，词嵌入 `257152×2048` FP32中NaN/Inf均0。报告 `/home/wuyan-lyj/thor/pi/artifacts/rtc-20000-audit-20260917-r1.json`，SHA256 `879e8a827e8eb0bad35505763040971c19dce785214c8b27e553e6abea08b8b6`。这只是源参数闸门，不代表动作输出精度通过。首次FP32转换入口在容器 `/bench` 挂载下因 `Path.parents[2]` 越界、尚未调用转换器而退出；已修正为存在repo路径时使用repo转换器，否则用同目录已审计转换器。重跑完成，产物 `/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-10h-20260916/20000-pytorch-fp32-r1`，811个映射张量/3353433872个元素与加载值逐位一致，`model.safetensors` SHA256 `4f4ceb6849b0739cfd68abc92e3d817864c7416ecae9fc3add83653beb9056b7`；输出FP32、无LoRA。`config.json`仅有Pi0.5结构字段，实际RTC训练上限10由同目录 `rtc_manifest.json` 绑定训练合同持有，不能把转换器打印的默认`rtc_training_max_delay=0`当成本保存点的训练语义。此阶段尚无JAX↔PyTorch动作对照/推理延迟或TensorRT验收，不删除原件或首次失败证据。

2026-09-17 用户指定旧推理还需约半小时测试，RTC 先只转换、不测试。现有 `export_pi05_rtc_onnx.py` 在真正写ONNX之前会加载同保存点JAX参考并在CUDA上跑9组 `d=0/1/10` 的eager/wrapper及JAX数值对照；`build_trt_engine.py`又要求这些数值门槛与动态前缀输入通过。因此本轮停在已审计的FP32权重，不绕过精度闸门伪造“已导出/已构建”，也不占用正在测试旧模型的GPU。待用户明确恢复RTC测试后，再生成JAX参考、导出ONNX和构建独立引擎；不自动替换100000服务。

## 2026-09-17 · 30000 保存点传至 Thor；100000 重载

用户允许从同一 RTC run 拉回 30000，并要求重载旧 100000 服务。服务器源目录为 `yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_10h_20260916/30000`；`_CHECKPOINT_METADATA` 含 `commit_timestamp_nsecs=1789623621937577783`，无同名未完成临时目录。Thor 目标为 `/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-10h-20260916/30000`。仅传 `params/assets/_CHECKPOINT_METADATA` 与同 run 的 `control/lego_pi05_rtc_base_10h_20260916/training_contract.json`；不传 `train_state`。六个大 OCDBT 文件经 Thor 5GHz Wi-Fi 从服务器并行续传，均退出0，最后整目录 rsync 补传退出0。`params/assets` 共16文件、`12440614997`字节，源/目标逐文件 SHA-256 清单一致；完成元数据 SHA256 `87477bac56c0bf9025d8891cab670870ad340e4bafee1d59b9c504fd38930e62`，norm SHA256 `b38ac082a729a4825c1a946c965183125382f10ccb5a75da89c2427019d1fefe`，训练合同 SHA256 `ef773631bb5dac8d4de055dc1d7161cf57c691b68ae50ec983e3a6297cb5ed5f`。norm/合同与 20000 相同，但参数文件身份按 30000 独立核对。此保存点**尚未做原始参数有限性审计、FP32转换或动作测试**；旧模型测试期间避免 CPU/内存带宽干扰，完成后从这份原件另建转换目录，不覆盖20000。

Thor `pi05-infer` 于 2026-09-17 14:01:15 +08 原容器重启，`--checkpoint`仍指向 `lego-full-20260913/100000-pytorch-fp32`、W/80 引擎仍为 `pi05-100000-trt-80-20260915-r1`，直连 `192.168.250.1:8000` 未改。重载后 `/healthz` 返回 OK，宿主 `nvpmodel -q` 为 MAXN，现有 `maxn_session.py -- docker wait pi05-infer` 会话仍在；这只证明服务就绪，不代替用户侧真实推理测试或跨 IPC 验收。

## 2026-09-17 · 30000 RTC 数值定位与 ONNX 导出

用户结束 100000 测试后要求停止旧服务、转换并启动 30000 RTC。旧 `pi05-infer` 已 `docker stop`，保留容器/权重/引擎；因旧 MAXN 包装器在容器此前重启后卡在 `docker wait`，精确终止该孤立包装器并按它保存的 clocks 配置恢复 120W。后续常驻服务不得继续用这一 `docker wait` 包装方式。Thor 原始 30000 Orbax 参数 51 个 FP32 leaf、非有限元素 0，审计 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-audit-20260917-r1.json`。FP32 转换目录 `/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-10h-20260916/30000-pytorch-fp32-r1`，811 张量、3,353,433,872 元素映射加载逐位一致，无 LoRA；`model.safetensors` SHA256 `fc60f64cfd05906edda9f446e113c159e1df6ece4ea479e27ba22b01791d5f60`，norm SHA256 `b38ac082a729a4825c1a946c965183125382f10ccb5a75da89c2427019d1fefe`。

JAX CUDA 参考需用独立 `openpi-pi:thor-trained-rtc-jax-ref-20260917-r3` 镜像；PyTorch 候选镜像只含 CPU JAX。JAX 镜像覆盖与训练代码快照 SHA 相同的 `gemma.py`，运行 `rtc_jax_reference.py` 时使用 `jax_default_matmul_precision=highest`。RTC 采样器原先在 `d=0` 直接调用普通标量时间采样；训练路径使用逐 token 时间 `[batch,50]`，已改为 `d=0` 同样走 RTC 时间分支。这个语义修正对首例输出仅约 `1.9e-6`，不是后述大偏差原因。JAX 参考为 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-jax-reference-20260917-r4`。

首次 Torch FP32 导出前数值门槛失败：首例 `d=0` 归一化最大绝对差约 `5.12e-4`、物理14D约 `3.42e-4`，而 Torch eager↔wrapper 完全一致；普通 Torch 采样也有同量级偏差，因此不是 RTC 前缀特有故障。根因是 `PI0Pytorch.__init__` 把进程级矩阵乘法精度设为 `high`，覆盖导出脚本在**模型加载前**设置的 `highest`。PyTorch 官方文档明确 `high` 可使用 TF32/BF16 内部计算，即使权重及输出张量仍为 FP32；不能以参数 dtype 推断实际计算精度。已使 `export_pi05_rtc_onnx.py` 与既有普通 Pi 导出相同：**加载模型后**设置 `torch.set_float32_matmul_precision('highest')` 并关闭 CUDA/cuDNN TF32。社区 OpenPI #810 报告过同输入转换前后输出不等，#958 指出 LoRA 静默丢失与 BF16 转存均会引入误差；本 30000 是全量 FP32，LoRA 故障不适用，实际根因由本地重跑确认。不可用社区的经验误差范围替代本项目数值闸门。

修正后输出 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-onnx-fp32-20260917-r4`：同噪声9个真实观测（`d=0/1/10` 各3个）全部通过归一化32D和物理14D各 `1e-4` 门槛；归一化最大差跨案例最高 `7.197260856628418e-6`，物理最大差跨案例最高 `6.039313729555573e-6`。Torch eager↔固定形状 wrapper 全案例逐位一致，ONNX 图检查通过，`previous_actions`/`prefix_mask` 保留为运行时输入，导出报告状态 `onnx_exported_engine_not_validated`；图文件 SHA256 `d8e855f7c415d00c4e3ee0d47ae9a4055774065dd64cb25af56a13f6718d4719`。这些是离线数值一致性，不是 TensorRT 或机器人物理任务验收。所有实际参考/导出计算在 MAXN 下完成，会话退出恢复 120W；构建/服务状态须另行查实时报告，不因本节推断已上线。

TensorRT 构建输出 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-trt-fp32-20260917-r1`：`--stronglyTyped --noTF32 --skipInference`，无量化，8输入/1输出，engine SHA256 `24bbf6329e6525d930d441dd85a01678be3d09cb4ffb6733ae20fac8ab304fc8`。`validate_pi05_rtc_trt.py` 用同9组真实观测/固定噪声、CUDA Graph对原始JAX物理动作：最大绝对差 `8.553785528997437e-6`，平均绝对差 `6.454800959446696e-7`，前缀逐位不变且全部有限，报告 `validation-synced.json`。第一版计时在GPU异步提交后立即读时钟，P50 `0.76ms`为**无效延迟**；已在 `TrainedRtcInference` 的采样调用后同步CUDA并重测，真实引擎调用 P50 `1028.23ms`、P95 `1726.60ms`（含首调用）；无权重污染或数值异常。该全FP32强精度路径是准确性基线而非低延迟最终方案。

Thor Docker `pi05-rtc-infer` 使用RTC候选镜像、`--restart no`。初次试运行曾占独立8001并取得 `ws-smoke-local.json`；用户随后明确要求同一个地址切模型，现已停止并保留旧8001容器为 `pi05-rtc-infer-8001-stopped-20260917`，重启新 `pi05-rtc-infer` 到固定 `192.168.250.1:8000`。`ss` 仅见8000监听，无8001。本机 `smoke_pi05_rtc_ws.py` 在固定8000对9例三路RGB/14D state/prompt与 `d=0/1/10` 请求全部成功，输出有限 `(50,14)` 绝对目标、已承诺前缀逐位不变；最终收据 `ws-smoke-fixed-8000.json`，MAXN下服务 P50/P95 `1033.02/1043.24ms`、往返 `1033.85/1044.67ms`。这不是跨IPC网络或闭环任务验收。宿主 `thor-pi-maxn-30000.service` 使用 `maxn_session.py` 包裹 `watch_docker_container.py`，在容器运行时维持MAXN；容器停止并超出短暂重启宽限后恢复日常120W。以后Pi系列切checkpoint只替换同一固定地址服务，不让3588按模型切端口。连接时应核对握手 `rtc_mode=trained`、`backend=tensorrt_cuda_graph`、checkpoint/norm指纹、`action_dt_s=1/30`、`action_0_relative_to_observation_policy_tick=0`；这一个0仅指训练数据行的30Hz policy tick，不是相机曝光到控制执行零延迟。3588 必须在同一目标tick上提交已承诺 `(d,14)` 绝对动作并只从返回 `actions[d]` 接管，`d` 限于0–10；晚到或前缀失配的回包必须丢弃。物理单位握手称关节rad、夹爪连续标称0闭/1开且不裁剪；真实硬件标定和曝光→policy tick偏移仍需现场核对，Thor不修改3588。

## 2026-09-17 · RTC 延迟与精度优化候选

用户要求争取约150ms、允许255ms作为候选，优先保留动作精度。训练时RTC不是推理时的梯度式inpainting；[Physical Intelligence后续论文](https://arxiv.org/html/2512.05964v2)明确说训练时前缀条件消除了推理时RTC的额外计算开销。本项目的 `d` 前缀仍是动态输入，10步Euler、H50、三相机和YAM逆变换均不更改。旧100000约106ms用BF16、文本桶80、时间缓存；首版RTC FP32约1033ms用200、无缓存，不能把差值归因于RTC本身。[NVIDIA精度说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)指出TF32保留FP32指数范围但尾数缩短，BF16尾数更短；二者不能以模型参数标签替代实际动作误差验证。

离线候选在Thor MAXN、同一30000权重/norm、9组真实YAM观测与原始JAX固定噪声下测试。稳态统计为9组样本再重复3轮，`server_infer`是采样器调用及CUDA同步，`server_total`还含Thor本机输入变换/逆变换，不含网络与控制端。首版验证只跑9例一次，首调用包含CUDA Graph捕获，不用其P95评估稳态。

| 路线 | 稳态采样P50/P95 | 稳态Thor处理P50/P95 | 物理14D对JAX最大绝对差 | 判定 |
| --- | --- | --- | --- | --- |
| FP32，TF32关，200 token，无时间缓存 | 约1028ms / 首轮P95含捕获 | 固定8000 WebSocket服务约1033/1043ms | 8.55e-6 | 高精度基线，太慢 |
| FP32权重＋TensorRT TF32，200 token，无时间缓存 | 247.90/255.12ms | 251.89/259.81ms | 0.0016182rad | 候选，不是已上线或闭环合格 |
| FP32权重＋TensorRT TF32，80 token＋时间缓存，10步 | 220.32/223.46ms | 224.01/227.37ms | 0.0017599rad | 当前同精度路线最快10步候选，仍高于200ms |
| BF16混合权重＋TensorRT，200 token，无时间缓存 | 133.72/137.67ms | 138.33/142.86ms | 0.0223596rad | 速度达标，但误差较大，暂不部署 |

TF32引擎 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-trt-tf32-20260917-r1`，SHA256 `c71d048b2c577b0070db904de6c9a8f3e43b451af38202980a15ca8f3a4ae974`，9例数值/稳态回执 `validation-steady.json`。BF16引擎 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-trt-bf16-200-20260917-r1`，SHA256 `d857ba8b6e44da1bbccca135dd51577bfc831c6e3f9990e22e3b1794025e4414`，回执 `validation.json`；BF16导出前Torch对JAX最大物理差0.03826rad，最终TRT最大0.02236rad，不能把这一路标成数值等价。以上两条均保持9/9有限 `(50,14)`、`d=0/1/10` 前缀逐位不变，只证明离线回放数值性质，不证明任务成功率。

FP32的80-token＋时间调制缓存预检 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-fp32-cache80-probe-20260917-r2`：9例eager↔wrapper不逐位相同，但原始JAX物理最大差6.54e-6、wrapper物理最大差1.33e-6，均通过数值门槛；按这个事实将优化版wrapper门槛单列为`1e-5`，不再错误声称bit-exact。正式ONNX `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-onnx-fp32-cache80-20260917-r1` 和TensorRT引擎 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-trt-tf32-cache80-20260917-r1` 已通过9例完整验证，收据为 `validation-detail.json`：物理14D MAE `9.12e-5`、P95 `2.93e-4`、最大 `0.0017599`（关节rad；最坏为case7、动作第49步、dim1），前缀保持且输出有限。`0.0017599`是所有标量中最大的一个，并非每个关节都如此。这一阶段均为离线实验；后续用户选择7步，已单独通过运行时/协议smoke并切换固定8000，见下节。

### 5/6/7/8/10 步采样配比实验（同一训练时 RTC checkpoint）

[训练时RTC官方实机实验](https://arxiv.org/html/2512.05964)使用5步去噪、H50、50Hz、训练延迟均匀采样0–10；本YAM训练也是H50与`d∈[0,10]`，但数据控制合同为30Hz，既有推理是10步。改变的是**推理 Euler 积分步数**，不改变训练权重、RTC最大前缀或动作空间；不能把官方50Hz、H100延迟和任务结果搬到Thor。`rtc_jax_reference.py`、RTC ONNX/TensorRT适配器现支持独立5/6/7/8/10步产物，原10步默认与线上路径不变；5/6/7/8步原始JAX、同checkpoint/norm、同9组真实观测、同噪声参考分别为 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-jax-reference-{5,6,7,8}step-20260917-r1`。比较程序 `scripts/thor/compare_rtc_step_schedules.py` 核对manifest与每例哈希后测量**采样设置差异**，不是对专家真值的误差：

| JAX原版同权重对比 | 全H50物理14D平均绝对差 | 全H50最大单元素差 | 每例前10个新生成动作最大差 |
| --- | ---: | ---: | ---: |
| 5步 vs 10步 | 0.0059225 | 0.0447544 rad | 0.0447544 rad |
| 6步 vs 10步 | 0.0040525 | 0.0318072 rad | 0.0318072 rad |
| 7步 vs 10步 | 0.0026847 | 0.0184450 rad | 0.0158643 rad |
| 8步 vs 10步 | 0.0016968 | 0.0122331 rad | 0.0107353 |

关节以rad计，夹爪为标称0–1连续值；混合14D的平均值不能被称为统一物理单位。5/6/7/8步最大关节差约2.56°/1.82°/1.06°/0.70°，但10步也只是另一种数值积分，不是真值。8步“前10个新生成动作”的最大项是夹爪值，不能标rad。原始回执在Thor `/home/wuyan-lyj/thor/pi/probes/rtc-step-sweep-20260917/jax-{5,8}-vs-10-r2.json`、`jax-7-vs-10-r1.json` 与 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-jax-reference-6step-20260917-r1/vs-10.json`。下一闸门是每个步数自身的JAX↔TensorRT数值差、MAXN稳态服务端时延与随后真机任务效果；不能只以这张表选线上精度。所有步数ONNX和引擎保留独立目录，不触碰固定8000服务。

同一 Thor/MAXN、FP32权重＋TensorRT TF32、80-token/时间缓存/CUDA Graph、9个真实三相机观测的离线候选结果（每行均与**同一步数**原始 JAX 对照，非与10步对照）：

| 去噪步数 | 采样器P50/P95 | Thor总处理P50/P95 | JAX→TRT最大单关节差 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 10 | 220.32/223.46 ms | 224.01/227.37 ms | 0.0017599 rad | 精度参考，慢 |
| 8 | 198.99/202.64 ms | 202.53/207.06 ms | 0.0010224 rad | 仅采样器P50低于200ms |
| 7 | 188.01/189.80 ms | 191.78/193.94 ms | 0.0008881 rad | Thor端低于200ms；加假定20ms余量约212ms |
| 6 | 182.92/183.44 ms | 186.53/187.26 ms | 0.0015228 rad | 加假定20ms余量约207ms |
| 5 | 174.54/175.09 ms | 178.23/178.85 ms | 0.0014782 rad | 加假定20ms余量约198ms，但采样设置偏离10步较多 |

5/6/7/8步各9/9输出有限且RTC冻结前缀逐位保持；本节表格是上线前离线测量，后续7步已切固定8000，其余候选未上线。全部均未做3588端到端/真机闭环。引擎和原始回执分别为 Thor `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-trt-tf32-cache80-{5,6,7,8}step-20260917-r1/validation-detail.json`；5步引擎SHA256 `f99dad5ef9985c93553db2fe09538fe261e80535f976a829d2e4015bbb957c59`，6步 `fe18dde29d6cee3ae1571fec30f4e4015a5f010b2103f021a2fb419d6d4c71db`，7步 `0b5cc53e2c4eaba58021b4018d471c3d8954633507d4d477088731f4a12455ce`，8步 `450f7793009696bc083a6d54b0254d5a995376959d7e4189d16185b348130fdd`。外部20ms仅为用户给的预算，不是已测3588往返。6/7步JAX→TRT物理14D平均差分别`9.06e-5`/`8.66e-5`（混合单位）、P95`2.92e-4`/`2.63e-4`、最坏均为case7/action49/dim1；这些也不是任务误差。若把外部20ms当硬预算，只有5步的P50约198ms勉强小于200ms，P95约199ms，余量极薄；6步约207ms，7步约212ms。控制侧真实端到端时延、排队抖动与真机效果未验证，不能宣称200ms闭环已达标。

### 2026-09-17 · 七步引擎切换固定服务

用户明确选择7步。`serve_pi05_rtc_trt.py --allow-validated-tf32-7step` 仅允许 `built_experiment_not_accuracy_validated`、FP32权重/TF32、未量化、strongly typed且`steps=7`的引擎；上线前读取 `validation-detail.json` 核对引擎SHA、checkpoint/norm绑定、JAX参考指纹、9例`d=0/1/10`有限输出/RTC前缀以及物理最大差≤0.002（本机实测0.0008881rad）。`RtcTensorRTAdapter`再次核验导出报告和引擎文件SHA。握手增加`precision_mode=fp32_weights_tf32_compute`，`denoising_steps=7`。22项相关代码测试通过。首次临时容器未指定`--entrypoint python`而退出126；显式加入口后恢复，没有触及旧8000。临时8001预检9/9通过后停止；旧10步容器改名 `pi05-rtc-infer-fp32-10step-preserved-20260917` 并停止，原12GB引擎及权重均保留。新 `pi05-rtc-infer` 用原地址 `ws://192.168.250.1:8000`、`--restart no` 运行；`docker inspect` 已核对实际7步引擎目录，8001无监听。

Thor本机正式8000在MAXN下9/9真实RGB三路、14D状态与prompt的协议回放全部通过，返回有限绝对目标`(50,14)`，`d=0/1/10`前缀逐位保持。回执 `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-trt-tf32-cache80-7step-20260917-r1/ws-smoke-fixed-8000-20260917-r1.json`：服务推理P50/P95 `194.26/204.31ms`、本机往返P50/P95 `195.03/205.41ms`；首请求往返`211.96ms`，后8例约`194–196ms`。这是短样本协议smoke，P95受首请求影响，不能代替稳态长测或3588↔Thor时延。握手权重SHA256 `fc60f64cfd05906edda9f446e113c159e1df6ece4ea479e27ba22b01791d5f60`，norm SHA256 `b38ac082a729a4825c1a946c965183125382f10ccb5a75da89c2427019d1fefe`，引擎SHA256 `0b5cc53e2c4eaba58021b4018d471c3d8954633507d4d477088731f4a12455ce`。动作/目标tick语义未变，客户端不要因7步改变RTC前缀协议；具体看[05](../../05_inference_and_rollout.md#2026-09-17--固定8000的-rtc-30000-服务)。

切换时旧容器停止让原MAXN会话结束，宿主短暂恢复120W；新容器启动后以临时`thor-pi-maxn-30000.service`重新执行原`maxn_session.py -- watch_docker_container.py --container pi05-rtc-infer`。18:39 CST实测unit active、`nvpmodel MAXN/0`、GPU 1575MHz及EMC 4266MHz锁频；停止当前容器后恢复120W。此unit用`systemd-run --collect`启动，不是持久开机服务；重启后需重新启动容器和MAXN会话。回退旧10步时必须先停止当前容器并释放8000，随后以旧名容器/原引擎恢复、重建对应MAXN会话并重新smoke；不能只把旧容器启动在另一端口、也不能同时占8000。没有读写3588；尚未证明闭环任务精度或全链路≤200ms。

### FP8 是独立的部署量化研究，不是改用 FP8 微调

2026-09-17 查阅 [Jetson AI Lab Thor 教程](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)、[openpi-thor 实现](https://github.com/xuweiwu/openpi-thor)、[NVIDIA TensorRT 精度说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html) 与 [量化方案](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)。主流 Pi0.5 Thor FP8 路线是**训练后量化（PTQ）**：保留高精度训练权重，用真实数据标定激活 scale，在 ONNX 图中插入明确的 Q/DQ，让 TensorRT 对适合的矩阵计算用 FP8；敏感计算保留较高精度。NVIDIA 教程在 LIBERO/H10/7D、MAXN 的不同合同下报告 FP8 TensorRT 约54ms、FP8+NVFP4约49ms；这些数字和余弦相似度不能外推为本项目 YAM/H50/14D/trained-RTC 的时延或物理关节误差。它也没有给出足以判定本任务闭环成功的逐关节/夹爪误差。FP8 的收益是更高矩阵吞吐和较小权重/激活传输，但精度代价包括舍入与范围截断；不是每个算子都改成 FP8，`--fp8` 字样也不代表已正确量化整个网络。

社区 `openpi-thor` 的 FP8 默认校准32个真实样本，保留 FP32 敏感岛与 strongly typed 构建；其 FP8 验证门槛 `min_cosine≥0.97, MAE≤0.08, max_abs≤0.3` 是该实现自己的数值合同，单位/动作空间与本项目不保证相同，**不得直接拿来放行 YAM**。其作者还记录更广泛的 NVFP4 曾在 TensorRT lowering 后把均误放大约30倍，现改为 attention-side NVFP4、Gemma MLP 留 FP8；说明量化范围比一个精度标签更重要。NVIDIA [PTQ/QAT 说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)区分：PTQ 不重训，QAT 才在训练时模拟量化误差；真正的 [FP8 混合精度训练](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/introduction/introduction.html)通常仍保存高精度主权重/优化状态，不能理解为只留下8位训练权重。当前没有授权或依据为 YAM 改训练路线。

若后续做 FP8 实机候选，应固定30000 checkpoint/norm、同一去噪步数/文本桶/RTC输入与噪声，使用 YAM 真实数据标定并单独导出；对同一步数原始JAX与未量化引擎比较完整物理 `50×14`、逐关节rad、双夹爪、每个 `d` 的新后缀前段、P95/P99/最坏值与稳态服务时延，最后由真机任务验证。FP8不得把更改步数造成的采样差异算作量化误差。目前只完成社区/官方调研，**本项目尚无 FP8 RTC 引擎、YAM 校准或测试数字**。
