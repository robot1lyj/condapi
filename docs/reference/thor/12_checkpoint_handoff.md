# 12 · 新 checkpoint → Thor 推理候选

## 默认路线与触发

### 2026-09-15 · 100000替代候选转换与离线测试完成

100000直传已于19:32:58 +08完成，退出0；params/assets共17个文件两端SHA-256全部一致。
Thor原始参数全量扫描非有限元素0，FP32转换成功，811个映射张量/3353433872个元素加载逐位一致。
原件在`/home/wuyan-lyj/thor/pi/checkpoints/lego-full-20260913/100000`，
转换目录为相邻`100000-pytorch-fp32`，已补齐自身assets（转换器现优先读checkpoint内assets，兼容父级回退）。
100000和158000的服务器目录构成均为params约12G、train_state约19G、assets约12K；
30多GB是完整续训检查点，Thor仅需params/assets，因此大小差别不是NaN原因。
精确`du -sb`统计100000参数12439691451字节（11.585GiB）、train_state19823844006字节（18.462GiB），
两项合计30.048GiB；原始字节证据见100000证据目录`server-directory-bytes.txt`。
`_METADATA`哈希在这两个模型中相同，仅说明结构元数据相同，不可当权重身份；用实际参数文件SHA清单区分。

同9组真实训练回放、同噪声/训练norm：JAX FP32与PyTorch FP32动作MAE
2.6340315733770177e-7、RMSE3.8481208646959624e-7、P95绝对差8.046627044677734e-7、最大差2.7418136596679688e-6。
完整数组形状9×2×50×14且全部有限；未做闭环/独立留出任务评估。
JAX参考`pi05-A-100000-20260915-r1`使用MAXN/锁频，结束恢复120W。
PyTorch首个完成诊断`pi05-100000-P-fp32-20260915-r2`实际为120W，不能当MAXN正式性能；
其历史result内base-model/benchmark-norm文案是脚本旧硬编码，实际suite和norm明确绑定100000，后续脚本已修正文案。
先前r1因未挂本地tokenizer缓存退出，保留失败，不安装联网依赖绕过离线要求。
200桶FP32时间缓存准备9/9与原采样器逐位一致，80桶FP32原严格诊断亦9/9通过。
80桶混合精度同桶封装9/9逐位一致；ONNX导出740.07s完成，新强类型非量化TensorRT引擎构建完成。
正式W回放`pi05-W-100000-20260915-r1`：9输入×5预热/20计时，共45预热、180正式调用，
P50 104.33888750412734ms、P95 105.05514599062735ms，输出9×20×50×14全部有限；
同引擎CUDA Graph/普通执行最大差0，重复输出差0。
相对本次JAX参考（相同9输入，2次计时）MAE0.0013109738136041589、RMSE0.0019055460306133575、
P95绝对差0.004256637394428253、最大差0.010577201843261719，均为原数据数值；归一化及逐维结果见JSON。
精度取每个observation首个正式输出，允许参考/候选计时次数不同，不把180调用当180独立精度样本。
W推理MAXN下GPU1573–1575MHz、最高55.5°C、板级输入功耗峰值144.616W；退出0并恢复120W/动态调频/自动风扇。
夹爪输出未额外裁剪，右侧本轮范围约[-0.000757,1.001333]，原始JAX也有略大于1的值；
不借此宣布机器人安全或任务效果已验收。没有部署常驻服务或操作3588。

Thor产物：`/home/wuyan-lyj/thor/pi/artifacts/pi05-100000-onnx-80-20260915-r1`、
`/home/wuyan-lyj/thor/pi/artifacts/pi05-100000-trt-80-20260915-r1/sampler.engine`（约6008.7MiB）。
新引擎、缓存、ONNX和结果均绑定本次FP32权重SHA
`f2ed6d5278d5579f025ffeb66c709fe0c58ce8988312cf322589247ebd2ad746`，原始权重身份另由17文件SHA清单证明。
复跑用scoped目录的`run_suite_host.py`，参数为
`--image openpi-pi:thor-pytorch-onnx-v6-20260907 --batch-id <新ID> --modes W
--engine-id pi05-100000-trt-80-20260915-r1 --checkpoint lego-full-20260913/100000-pytorch-fp32
--suite pi05-100000-replay-20260915/suite.json --warmups 5 --repeats 20 --code-commit <实际代码版本>`，
Thor宿主sudo执行，使用新ID且从日常120W开始；不可覆盖既有实验。
下载完成后的10分钟监控已删除，不再声称仍在下载。
证据：[100000中文报告](../../reports/thor/100000.html)、[100000目录](../../reports/thor/evidence/20260915/100000/)。

### 2026-09-15 · 158000首次实际接入结果

后续只读追查：154000/156000/158000完整词嵌入均为相同9个NaN；
坐标为(117501,628/629/631)、(117502,244/1909/1911)、(117503,1525/1526/1527)。
保留点定点抽查中100000这三行正常，110000已有9个NaN，120000/140000亦相同；
首次异常在已检查保存点中界定为100000→110000，非129501/134051的后续nonfinite事件首次引入。
`scripts/train.py`的param_norm明确排除input_embedding；日志指标有限不能保证该权重正常。
152000/158000对应三行优化器mu/nu均有限。尚无证据区分GPU/内存错误、训练更新或保存原因。

2026-09-15 18:57 +08:00：按用户明确授权，删除Thor上的
`/home/wuyan-lyj/thor/pi/checkpoints/lego-full-20260915/158000`（约12GiB），
服务器原件、审计JSON、失败容器日志和旧基础模型产物均保留，可从服务器重新下载。
只删除这个明确目录，不删除其他缓存/测试数据。开始直传替代候选100000：
源`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b32_xid13_recovery_20260913_nccl_96261_hl9/100000/`，
目标`/home/wuyan-lyj/thor/pi/checkpoints/lego-full-20260913/100000/`。
仍只取params/assets/完成元数据，排除续训train_state；完整检查结果不由定点正常推断。
Thor日志`/home/wuyan-lyj/thor/pi/logs/download-100000-20260915.output.log`，
`.started.json`记录命令，结束后`.exit.json`记录退出码；当前观察388302913字节，尚未下载完成。
既有每10分钟监控`thor`已改为100000并启用，优先Wi-Fi管理；完成须退出证据及文件核对。
100000全参数只读扫描未取得完整报告；分块重试输出前33个张量均有限后，SSH退出255，
不能把部分扫描写成全模型通过。后续在下载完成的Thor本地执行原始参数审计，避免远程检查会话中断。

用户限定今天只做158000转换和测试，不部署服务、不联调3588、不拉其他checkpoint。
源端与Thor的16个params/assets文件SHA-256逐项一致，norm SHA为
`044aad51d9439e4b69dcf87c2d0f5b22ee077bbd83acf785c034b598fc4d4dcc`。
实际FP32转换在词嵌入参数有效性检查处退出1，未生成输出目录。
随后在同一Pi v6镜像、Thor CPU上用原生restore_params(dtype=None)只读审计原始JAX：
`PaliGemma/llm/embedder/input_embedding`（257152×2048、FP32）含9个NaN，
全部参数合计9个非有限元素，无Inf。源metadata SHA为
`2912d15ba42fe0d742b5ac46e88804dacce69644369b4cf9b7405cf7d8fe7247`。
这不是BF16/引擎舍入，尚未定位训练原因或受影响token；禁止静默清零后声称保真。
因此未继续导出、构建引擎或推理测速。全程120W，无MAXN会话；容器GPU警告来自有意使用CPU转换，真正失败为参数检查。

复用入口：`scripts/thor/prepare_checkpoint_suite.py`将不可变真实输入绑定checkpoint自身norm，
`scripts/thor/audit_checkpoint_finite.py`对原始参数输出逐张量shape/dtype/NaN/Inf计数。
Thor scoped脚本目录为`/home/wuyan-lyj/thor/pi/probes/finetune-158000-20260915`；
回放为`/home/wuyan-lyj/thor/pi/test-data/pi05-158000-replay-20260915`，旧回放不修改。
转换容器`pi05-convert-158000-20260915`退出1，审计容器`pi05-audit-158000-20260915`退出0，保留日志。
证据与状态页：[158000报告](../../reports/thor/158000.html)。当前阻塞不通过扩大容差解决；
换用有效checkpoint或另行获准定位/修复源权重后，才重新创建run继续。未操作训练或3588。

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
核对源目录层级及 assets 位置；转换器优先checkpoint内assets、缺失时兼容父级，仍须核对实际norm身份。
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

**宿主封装参数：**`run_suite_host.py`和`run_export_host.py`已支持
`--checkpoint <相对checkpoints目录>`、`--suite <相对test-data目录>`；默认仍为基础模型，仅供旧复测。
新checkpoint必须显式指定两项；本地tokenizer通过`OPENPI_DATA_HOME=/cache`读取，禁用TF32覆盖。
`run_suite_host.py`还支持`--warmups`/`--repeats`，默认5/20。
轻量命令构造/路径越界测试见`checkpoint_host_test.py`。每阶段复用MAXN恢复包装器，
显式挂载本次权重、suite、norm和结果目录。
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
