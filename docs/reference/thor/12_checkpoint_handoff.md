# 12 · 新 checkpoint → Thor 推理候选

## 默认路线与触发

2026-09-11 用户确认保留现有 W 路线，并要求新 checkpoint 优先自动接入。
技能源码为 `skills/thor-checkpoint-deploy/`，安装到本机 Codex skills 目录供自动选择。
这是 agent 收到新权重交接后的工作流，不是已安装的目录监听器或自动生产发布器。
本文件为 [08](../../08_thor_edge_deployment.md) 下属操作手册；实测、版本与局限分别查
[执行记录](10_acceleration_execution.md) 和 [固定候选](11_pi05_candidate_test_plan.md)。

固定路线：**Pi0.5 全量微调 JAX FP32 → 审计转换 PyTorch FP32 → BF16 主计算＋FP32 敏感部分
→ 固定十步 FP32 时间条件缓存 → 非量化强类型 TensorRT → CUDA Graph**。
短文本 W/80 桶优先，V/200 桶保留；不改三相机、H50、十步或 YAM 14D 语义。

## 1. 一次收齐交接输入

- 已写完且可完整恢复的 checkpoint 目录和 step；结合保存完成标志/训练器写入规则确认，不读取仍在写的 latest。
- 同次训练配置、norm stats、tokenizer 身份及代表该任务的本地 observation 回放。
  复用既有 tokenizer 前核对身份；不能拿基础模型测试 norm 冒充本次训练 norm。
- 原始权重清单/哈希、配置/norm/回放指纹；识别加载的是 policy/EMA 权重而非优化器状态。
- 默认全量微调；若实际发现 LoRA、不同结构或非 FP32 权重，明确不属于当前已审计输入，不静默丢弃或向上转型冒充 FP32 原件。

用户给出路径后，优先按已有项目配置寻找配套文件，只问缺少且无法定位的必要信息。
USB 管理链路可用时优先传输；否则用已有 SSH。只传新文件，传完核对来源身份。
来源不在已授权机器/路径范围时先取得方向。禁止后台持续从服务器流式取样。

## 2. 每个 checkpoint 独立产物

为本次建立带 step/短哈希的唯一 ID；原件、FP32 转换、导出、引擎、回放输出分别写新目录。
保留原始 JAX、norm、旧候选和日志；不用覆盖旧文件、重命名新权重为旧基础模型来绕过脚本限制。
复用现有 Pi 系列镜像，在容器内做模型相关处理，记录实际镜像 ID、代码版本及执行命令。
同 checkpoint 的失败重试也用新 run ID；只有输入/代码/精度/环境指纹都匹配的完成阶段才可复用。

## 3. 转换与参考

复用 `adapters/openpi/convert_jax_model_to_pytorch.py`。在系列容器内指定本次路径：

```text
python adapters/openpi/convert_jax_model_to_pytorch.py
  --checkpoint-dir <本次原始checkpoint>
  --config-name pi05_yam
  --output-path <新建FP32目录>
  --precision float32
```

上面是参数示意，执行时组成一个命令；入口 CLI 以当前源码/--help 为准。
**precision 必须显式 float32**，转换器默认 bfloat16 不适合保真中间产物。
核对源目录层级及 assets 位置；转换器现从源目录父级查 assets，不能仅凭成功退出认定 norm 已随附。
保留转换审计：参数覆盖、预期例外、dtype 和加载值一致性。

用同次权重生成 JAX FP32 golden，并与新 PyTorch FP32 输出对照；固定输入、噪声、预处理与 norm。
复用 `benchmark_pi05.py` / `benchmark_suite.py` 的明确参数入口。
不能对旧基础模型的 golden 比新微调模型，也不能把训练 loss 当转换精度。

## 4. 新缓存、导出与引擎

复用 `scripts/thor/export_pi05_onnx.py`：
`--checkpoint <本次FP32路径> --suite <本次suite> --output <新目录>
--compute-dtype bfloat16 --cache-time-modulation --text-bucket 80`。
先完成当前导出器需要的 prepare/padding 对照，按实际接口提供证据，不禁用其检查。
时间缓存必须从本次权重按原 FP32 时间递推和原 batch=1 投影重新计算；不是跨 observation 的视觉缓存。
权重、时间表或步数变化必须重建缓存及 engine。

tokenizer 接口保持 200，只裁掉 mask=false 的尾部；先核对本次任务状态/prompt 的有效 token。
超过 80 时明确选择同 checkpoint 的 200 桶导出/引擎，不截断真实 token。
当前服务层自动桶路由未实现，不声称已有自动回退。

用 `build_trt_engine.py --source <新导出目录> --output <新engine目录>` 构建，
保持 strongly typed/noTF32，无 FP8/FP4、FP16 修补或非有限值截断。
engine 必须在目标 Thor/兼容容器环境构建；记录 ONNX 外部权重及引擎指纹。

**现有宿主封装的适用边界：**截至本次写入，`run_suite_host.py` 和
`run_export_host.py` 仍硬编码基础 checkpoint 与旧 suite。
它们的现成命令只用于基础模型复测，不能原样当新权重流水线。
新 checkpoint 首次接入时，使用其容器/电源包装结构调用上述已参数化的底层入口，
显式挂载本次权重、suite、norm 和结果目录；或先对封装补相应参数并做轻量命令构造测试。
不整体同步/替换远端主检出，不复用与当前代码不一致的镜像内旧转换器。

## 5. 最小有效测试与交付

先取任务真实回放中的少量代表输入，检查加载、finite、形状及转换误差；无异常后做正式回放。
可沿用 3 条轨迹各早/中/晚共 9 输入、各 5 次预热＋20 次正式调用的规模，
但样本必须适合本次任务，180 次计时不代表 180 个独立精度样本。

分层保留：JAX↔PyTorch FP32；同桶 eager↔缓存 wrapper；导出前混合精度↔TRT；
JAX↔最终输出；同 engine 图↔非图。最后两者不得混淆。
报告 MAE、RMSE、P95绝对误差、最大绝对误差与逐关节/夹爪分布，
保留归一化及数据集单位两种口径。物理单位未审计就不换算成毫米或角度。
历史基础模型的约 0.0025 MAE 是参考观察，**不是新 checkpoint 通用通过阈值**。
若已有任务容差，测试前记录并按它判定；否则交付数值结果并标明任务精度未评估。

完整调用 P50/P95 包含 Thor 预处理、同步推理及后处理，不含 3588/网络。
只在需要推理测量的会话内用 `maxn_session.py` 临时 MAXN，记录频率/温度；
结束核对恢复 120W、动态调频和自动风扇，不将 MAXN 设为开机默认。

生成本次中文 HTML 及 manifest：checkpoint→norm/config→转换→导出→engine→结果证据相互可追溯。
交付新产物目录、实际可复跑命令和状态：
“已构建”“已完成真实离线推理”“数值门槛通过/失败/未定义”“任务效果未评估”分别说明。
不因任务容差暂缺而反复暂停普通离线处理；也不宣称生产服务或闭环已验收。
服务切换需单独授权，旧路径和 RTC 关闭/回退能力保留。
