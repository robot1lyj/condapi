---
name: thor-checkpoint-deploy
description: Convert and validate full-finetuned Pi0.5 JAX checkpoints for Thor with two routes, ordinary inference or training-time RTC. Use for new checkpoint handoffs, conversion, or Pi deployment; select the route from user intent and the training contract. Not for training or other VLA families.
---

# Thor checkpoint 接入：普通 / RTC

收到新 checkpoint 后沿用已验证路线，完成转换、构建和真实回放。自动选用本技能不等于后台监听，也不自行替换线上模型。项目内 `skills/thor-checkpoint-deploy/` 是技能源码，安装副本随源码同步；设备状态和实验结果仍由项目 docs 持有。

## 先确定路线

以实际 Git 根目录定位 condapi，遵循 AGENTS 和 docs/cache 路由。优先使用用户指定的“普通”或“RTC”；未指定时读取配套训练合同/配置：`rtc_training_max_delay > 0` 走训练时 RTC，明确为0走普通。合同缺失时先收齐已完成权重、norm、tokenizer及配置，只对无法从产物确认的关键选择询问。不能仅凭目录名、相同参数形状或转换器生成的默认 `rtc_training_max_delay=0` 判定路线。

| 选择 | 转换/推理路线 | 按需读取的项目 owner |
| --- | --- | --- |
| 普通 | JAX FP32保真转换 → W的BF16主计算/FP32敏感部分与时间缓存 → 非量化TensorRT + CUDA Graph | `docs/reference/thor/12_checkpoint_handoff.md`，优先第1–5节操作流程 |
| RTC | JAX FP32保真转换 → FP32权重、TF32 TensorRT、时间缓存 + CUDA Graph；当前已验证配比为7步 | `docs/reference/thor/13_trained_rtc_inference.md`，先看最新状态及分位数修正，再看工具流程 |

普通 checkpoint 不能仅加一个推理开关就冒充训练时 RTC。用户明确要求在 RTC 权重上关闭条件化时，保留普通路径并单独验收，不能静默改变其指定模式。RTC分支不套用普通W的BF16精度结论；7步是已有配比，新checkpoint仍需独立对照，不重新扫描全部历史候选。

## 共用接入流程

- 集中核对完整保存状态、实际参数文件身份/有限性、模型配置、checkpoint自身norm和本地tokenizer。只取推理必要产物；原始权重与失败证据保留。
- 在Pi系列容器内操作；每个checkpoint使用新的产物目录，缓存、ONNX、engine、参考输出和回执绑定本次权重/norm/代码/路线/采样步数。路径、镜像和阈值从owner及实际代码确认，历史例子不是当前启动命令。
- 先用少量真实三相机、14D state和prompt做同输入同噪声对照，再正式计时。输出按原policy逆变换恢复为有限 `(50,14)` 绝对目标；关节rad、夹爪连续值的合同以数据审计为准，分别报告误差。
- 推理、预热和性能测试前进入MAXN/锁频；会话结束恢复120W。拷贝/转换准备不常驻MAXN。获准保持在线的服务由MAXN会话跟随，停止服务后恢复日常模式。
- 只在授权来源取已完成产物，不启动训练，不操作3588。复用现有同一Pi服务地址；转换/离线测试请求不隐含切换线上模型，已获上线授权则完成加载、握手与本机真实协议测试，不反复请示。

## 普通分支

复用 `adapters/openpi/convert_jax_model_to_pytorch.py`、`scripts/thor/prepare_checkpoint_suite.py`、`export_pi05_onnx.py` 和普通TensorRT构建/回放入口。先核对原JAX与转换后FP32，再测试本权重的W混合精度与缓存；不能复用其他checkpoint的缓存或参考结果。服务入口 `serve_pi05_trt.py`，协议测试 `smoke_pi05_ws.py`，RTC关闭。文本桶保留全部有效token，超桶显式处理，不静默截断。

## RTC分支

复用 `scripts/thor/prepare_rtc_checkpoint.py` → `prepare_rtc_cases.py` → `rtc_jax_reference.py` → `export_pi05_rtc_onnx.py` → TensorRT构建 → `validate_pi05_rtc_trt.py`。所选步数在JAX、Torch、ONNX、引擎、服务中一致；FP32参考关闭TF32，候选引擎显式记录TF32。运行参数以当前脚本帮助和owner为准。

### 前缀语义与共同错误防护

- Pi0.5/YAM训练使用分位数归一化。绝对前缀先按**新观测state**把12个关节变为delta、两夹爪保持absolute，再按训练配置的 `use_quantile_norm` 编码，补齐32D。JAX参考和Torch/TRT服务都必须显式传 `use_quantiles`，不能落入均值/标准差默认值。
- 独立核对前缀编码与训练数据transform。两个后端调用同一错误编码器也可能数值接近；返回前缀逐位保持不能证明模型内部条件正确。分位数修正前的RTC报告不作为新验收依据。
- `committed_actions` 为已承诺的 `(d,14)` 绝对目标，`0 <= d <= 训练dmax`；`observation_policy_tick == target_start_tick == committed_start_tick`，均为30Hz policy tick。相机时间不能直接替代目标tick；d=0提交空前缀。
- 模型内前缀token保持干净动作/flow time 0，只去噪后缀；每请求更新动态prefix与mask，固定张量形状仍可使用CUDA Graph。保持RTC功能，不隐式退回普通采样或guidance RTC。

### 验收与上线

覆盖真实 `d=0/1/dmax`，对比原JAX与候选的完整物理输出、逐关节/夹爪误差，并检查 `action[d-1]→action[d]` 交界和新后缀相邻步进。现场事故回放可复用 `prepare_rtc_incident_cases.py`、`replay_rtc_incident_trt.py`；原噪声/请求未记录时说明重建局限。

服务用 `serve_pi05_rtc_trt.py`，绑定本次正确前缀验证回执与JAX参考（`--validation`、`--jax-reference`），握手核对 `rtc_prefix_norm=quantile`、步数、实际后端和权重/norm/engine指纹。当前服务保留 `0.2 rad/tick` 粗大关节跳变拒绝；检查广告阈值与实际行为，不能为通过测试静默放宽。超限返回错误、不返回动作、不退回普通推理。它不替代控制侧保护或任务精度评估。

用 `smoke_pi05_rtc_ws.py --expected-prefix-norm quantile` 做本机协议验证；默认覆盖0/1/10，其他训练上限或事故d=9用 `--required-delays` 指定实际集合。跨IPC连通、端到端时延、用户真机效果分别记录；单次用户“效果好”不能外推为所有checkpoint或成功率。

## 交付与失败处理

报告给出所选路线、checkpoint及配套norm身份、产物路径、精度/步数、P50/P95与异常尾延迟、误差、前缀/后缀检查、服务和电源状态。数值门槛按owner区分后端误差与真实动作步进，不根据本轮结果临时调宽。

遇到不完整checkpoint、非有限参数、映射或合同冲突时保留证据，定位实际失败层；缺少机器人任务容差仍可完成离线误差/时延报告。没有新checkpoint时只更新技能/流程，不重转换旧权重或重启现有服务。
