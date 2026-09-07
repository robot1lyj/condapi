# Pi0.5 当前候选与后续测试方案

## 2026-09-08 当前选择：W，完整调用约 104 ms

本轮固定 **W：原始 JAX FP32 → 已审计 PyTorch FP32 转换 → BF16 主干 / FP32 敏感运算 → 固定十步时间条件 FP32 缓存 → 非量化 TensorRT → CUDA Graph**。短文本使用经过 mask 检查的 text80 计算桶；tokenizer 接口仍为 200，三相机 224RGB、全部真实文本/状态 token、H50、去噪 10 不变。

9 个真实输入 × 20 次正式调用：**P50 104.25 ms / P95 104.94 ms**，相对原 JAX A 的动作 MAE 0.002504、最大差 0.017282 数据集单位，归一化 14D 最大差 0.025594。比 V 节省 4.59 ms；本样本汇总误差未扩大，不等于任务效果改善或无损。单纯图捕获对同引擎非图输出 9/9 零差异，重复输出零变化。GPU 最高约 56.3°C，测试结束恢复 120W。

这是“100ms 左右”的**部署候选**，尚未达到严格 ≤100ms，也不是生产服务/任务精度已验收。完整计时含 Thor policy 预处理、推理、后处理和 GPU 同步，不含 3588 采集/网络传输；本轮不涉及 3588。

### 固定产物

- Pi 系列候选镜像：`openpi-pi:thor-pytorch-onnx-v6-20260907`，基于 NVIDIA `nvcr.io/nvidia/pytorch:26.05-py3`；Docker ID `sha256:89d40707e81ab1d18078aba0471c65e31af201d95b03fe6c13bab129ebff221b`。不在宿主安装模型依赖。
- TensorRT 10.16.1.11 / PyTorch `2.12.0a0+5aff3928d8.nv26.5.50603568`；`stronglyTyped`，TF32 关闭，没有 FP16 修补、FP8/FP4 或 `nan_to_num`。
- Thor 引擎：`/home/wuyan-lyj/thor/pi/artifacts/pi05-trt-text80-20260908-r1/sampler.engine`，SHA-256 `a0052cf5847f565d804fe633f4445cd58037bf19e094704286986498775b1e78`。
- 来源导出 `pi05-onnx-text80-20260908-r1`，ONNX SHA-256 `715e674d8b1104ecdfd012f29d5bdbe4c7bb5050cb3003f76fb9a6d6034f2d76`；完整权重、噪声、norm、输入和代码指纹见 export/engine/result JSON。
- 性能运行 `pi05-W-20260908-r1`，运行代码 `849c8c884b8e21e5a6c6c580545c4269091c8180`。所有原始 JAX、FP32 转换权重及旧引擎保留。

### 一条复测命令

在 Thor 仓库目录执行，批次名必须新建；脚本自行临时 MAXN/锁频并在退出恢复日常设置：

```bash
sudo python3 scripts/thor/run_suite_host.py \
  --image openpi-pi:thor-pytorch-onnx-v6-20260907 \
  --modes W --batch-id manual-001 \
  --engine-id pi05-trt-text80-20260908-r1 \
  --code-commit "$(git rev-parse HEAD)"
```

复测先检查退出成功、180 次正式调用、P50/P95、finite、重复差和同引擎图/非图差，再运行 `compare_suites.py` 对 JAX A；报告继续保存所有误差，不把 observed error 自动当成容差。当前无需重新下载镜像、权重或构建引擎。超过 text80 的有效输入显式使用 V 的 200 桶（见下节），**自动路由尚未实现**。

下一阶段优先用实际微调 checkpoint 完成 LoRA 合并与任务精度验证，再决定是否需要继续优化；本轮不为再省几毫秒默认降位量化。当前基础模型没有 LoRA，不能宣称该路径已经验证未来 LoRA 模型。

本文持有候选复测顺序与下一轮测试规则；逐次成绩、误差和失败原因由 [加速执行记录](10_acceleration_execution.md) 持有，安装与容器环境按 [Thor 部署](../../08_thor_edge_deployment.md)。本方案只操作 Thor，不读写 3588。

## 原 V 可复现路径（长文本保留）

原始 JAX FP32 checkpoint → 已审计的 PyTorch FP32 转换文件 → Pi 系列 v6 容器内 BF16 主计算、保留 FP32 敏感部分 → 原 batch=1 FP32 固定时间条件投影缓存 → 非量化 ONNX → strongly typed / noTF32 TensorRT 引擎 → CUDA Graph → 原 YAM 输入/归一化/动作还原。

当前输入不变：三路 224RGB、文本上限 200、H50、10 步去噪、内部 32D/输出 14D。缓存绑定固定权重、时间递推和步数；修改任一项都必须重建。原始权重、LoRA、norm 和 JAX 参考不得被引擎或缓存替代。

当前选择 V 作为性能候选，不作为任务精度已通过的生产默认。保留 JAX FP32 参考和较简单的 PyTorch I 路径；基础模型未针对 YAM 微调，不能承诺实机成功率。

## Thor 上复测 V

在 **Thor 的项目目录**执行，`batch-id` 每次改为未用过的值；重复 ID 会拒绝覆盖。下例使用已构建的、指纹可核验的引擎，不重建系统或下载权重：

```bash
cd /home/wuyan-lyj/condapi
sudo python3 scripts/thor/run_suite_host.py \
  --image openpi-pi:thor-pytorch-onnx-v6-20260907 \
  --modes V \
  --engine-id pi05-trt-timecache-20260907-r1 \
  --batch-id 20260908-verification-01 \
  --code-commit "$(git rev-parse HEAD)"
```

入口自动开启本次测试的 MAXN/最高频率并在退出后恢复日常模式。容器断网、读取 Thor 本地录像与权重，不依赖训练服务器实时传输。确认 run 的 exit 为 0、power-after 为 120W，再接受该次数据。

正式延迟包含输入处理、模型执行、同步等待和 YAM 动作输出，不包括 3588 采集/机械臂执行或生产以太网传输。原始耗时、动作数组、噪声、引擎/代码/输入指纹、温度与频率记录分别保留；Profiler 诊断不能混入正式延迟成绩。

## 精度对照不能混为一个“通过”

1. JAX → PyTorch 转换：保留 FP32 原始产物，映射/未映射项按转换审计解释，不默许 LoRA 丢失。
2. 包装器/缓存准备：与原采样器同输入、同噪声对照，当前固定时间缓存已在 9 个真实输入上逐值一致。
3. 引擎执行：同时比较 JAX FP32 和导出前 BF16 eager，记录归一化及数据集单位误差；不能仅凭 finite 通过。
4. 图重放：逐输入与**同一引擎**非图执行要求完全一致。此结论不外推到不同引擎。
5. 任务效果：微调后使用该 checkpoint 的 norm、实际任务数据与成功率验收；当前没有物理单位容差或闭环放行。

## 下一项：只消除被 mask 屏蔽的文本填充

当前真实样本只使用 64–70 个文本 token，但模型仍处理 200 个位置。这是待测候选，尚未改变当前部署合同。

- 保持 tokenizer 原上限 200，先生成完整 token 和 mask；只允许去除末尾全部 mask=false 的位置，禁止截断有效 prompt 或状态 token。
- 先用原始 FP32 权重比较完整与候选路径，排查位置编码、注意力掩码、KV 长度和 suffix 位置是否改变语义；FP32 对照不合格则停止该候选。
- 再测试 BF16/FP32 引擎。尺寸变化可能改变内核舍入，须分别记录准备误差和最终对 JAX 的误差，不把“数学上被 mask 屏蔽”写成未经测试的逐位一致。
- 任何超出候选文本桶的输入，必须选择原 200 路径或明确拒绝；不静默截断。生产服务是否已具备回退须另行验证，不能把计划当实现。
- 保持三相机、H50 和十步不变，正式计时与 V 使用同一批真实输入。目标仍为约 100 ms 或更低，不仅比较模型内部异步启动耗时。

若该候选没有明确收益，则保留 V 并依据层级数据选择其他融合/局部精度实验，不无限重复调度微调。FP8 只作为独立、有精度对照的后续方案，FP4 不作为默认。

## 2026-09-08 · 文本填充候选的独立诊断

已完成三个独立运行：`pi05-prepare-text200-fp32-20260908-r1`（FP32 控制）、`pi05-prepare-text80-fp32-20260908-r1` 和 `pi05-prepare-text80-bf16-20260908-r1`。控制组 9/9 完全一致；text80 FP32 最大原始输出差 2.712e-6，原先设定 `rtol=1e-5, atol=1e-6` 的诊断只有 **8/9** 通过，失败发生在有效的右夹爪维度，不是被丢弃的 32D 尾部。BF16 最大原始输出差 0.00279969。不能把这三次准备诊断算作新的正式测速配置，也不能称为原 200 桶零误差。

继续的是**离线实验，不是精度放行**：另用 PyTorch 已有的 FP32 默认比较容差检查数值尺度，保留原严格失败记录；这不是事后修改任务容差。实际容器的默认容差、原报告及数组 SHA 写入新导出证据。[官方测试接口](https://docs.pytorch.org/docs/2.14/testing.html) 的浮点比较默认值不代表机器人任务容差；[数值说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) 也不保证等价尺寸变换逐位一致。

导出时分开两条对照：①同 text80 的旧 PyTorch 采样器对新缓存 wrapper，仍要求 9 输入完全一致；②text80 对原 200 桶的漂移单独保留。最终 TensorRT W 必须再次对原 JAX A 比较，不能只和已改变文本尺寸的中间参考比较。相机、状态、真实 token、H50、去噪 10、原始 FP32 权重和固定噪声全部不变。

`TextBucketSampler` 与 TensorRT 入口检查 **80 之后任何有效 mask**，有则明确拒绝，不静默截断。当前未实现服务层自动路由；长输入须显式选原 V 的 200 桶路径。CPU 测试验证注意力边与位置映射一致、原 observation 不被改写、长输入在引擎执行前拒绝。候选不默认替换 V。

## 未来 JAX LoRA

当前基础 checkpoint 没有 LoRA；现有转换器拒绝静默丢弃 LoRA。微调后先实现并验证原生 JAX 下的 FP32 合并（合并前后同输入对照），再执行转换、缓存和引擎测试。未完成这一环节前，不能把本基础模型路径称为已经支持训练后的 LoRA 部署。
