---
name: thor-checkpoint-deploy
description: Prepare newly supplied Pi0.5 full-finetuned JAX checkpoints for Thor inference using the condapi BF16/FP32 TensorRT route. Use when a checkpoint is ready, training outputs are handed over for inference, or the user asks to convert or deploy a new Pi checkpoint; not for training or other VLA families.
---

# Thor checkpoint 接入

用户交付新 checkpoint 或要求接入推理时，优先完成既定路线，不重新开展框架选型。默认自动选择本技能；这不是后台目录监听器，不创建轮询任务，也不自动替换线上服务。

## 找到项目与唯一事实来源

以实际工作区的 Git 根目录定位 condapi；若当前目录不是项目，先定位已知项目，不向无关仓库写文件。遵循 AGENTS 和 docs/cache 路由。
执行前读取项目 `docs/reference/thor/12_checkpoint_handoff.md`；它持有实际接入步骤和工具入口。镜像、设备及历史数值仅按该文件引用的 owner 查询，不复制为技能内长期状态。若缺少项目或必要输入，只询问缺失项，不猜 checkpoint/norm。

## 默认行为

- 面向 Pi0.5 全量微调 JAX/Flax：FP32 保真转换 → BF16 主计算/FP32 敏感部分 → 新权重的 FP32 时间缓存 → 非量化 ONNX/TensorRT → CUDA Graph。不擅自切换 Dexmal、FP8、FP4、LoRA 或训练框架。
- 收到路径即推进完整的离线接入任务：确认写入完成、定位配套配置/norm/tokenizer、本地化数据、转换、构建、真实回放、中文结果。一次集中核对缺失输入；正常阶段不用反复请示。
- 复用 Pi 系列容器；新 checkpoint 使用新的不可覆盖产物目录。不要重装系统、污染宿主或为每个 checkpoint 建新系列环境。
- 所有缓存、ONNX、engine 和 golden 都绑定本次 checkpoint 身份；旧基础模型成绩不能充当新模型验收。
- 只操作获准的 Thor 和本地项目。训练服务器只在本次已授权来源范围内取已完成产物，不启动训练；不读取或改动 3588。
- 仅测试/执行推理期间临时 MAXN/锁频，复用恢复包装器；转换、拷贝和空闲不设常驻 MAXN。
- 完成的是“可加载并已实测的离线候选”。生产服务切换、跨 IPC 控制与机器人任务放行分开处理，不靠数值相似自动批准。

## 效率与停止条件

先做小而真实的同输入/同噪声对照，再做固定配置正式计时；不用每次重跑所有历史候选或无限加严门槛。只对实际失败层定位。
参数缺失/映射失败、非有限输出、实际权重身份不符或没有完成的 checkpoint 时，保留产物并报告具体问题。没有任务容差时可以继续产出离线误差和延迟报告，但不能自创容差宣布机器人可用。
报告给出 checkpoint、产物目录、精度、P50/P95、主要误差、通过/未通过/未评估状态以及恢复电源状态。没有新 checkpoint 时只完成技能/流程准备，不运行旧模型冒充新接入。

