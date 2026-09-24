# 04 · Mode — Model Operations

仅在 YAM 数据、Pi 训练、评估或 DAgger 操作需要额外边界时读取；不得代替 `AGENTS.md` 或当前实验 owner。

- 当前 Pi 默认路线是 `pi05_yam` **全量微调**；LoRA 属历史或另行批准的实验，不自动回退。训练配方、checkpoint 与续训条件从 [03](../../03_training_and_evaluation.md) 当前适用章节取，数据与单位从 [04](../../04_data_contracts.md) 取。
- 先核对数据版本、任务 prompt、三路图像、14D 原始 state/action、单位、split 和对应模型的 norm/processor。Pi 内部 32D/H50、关节 delta/夹爪 absolute 不套到 XR-1 或其他模型。
- 训练只能在获准服务器计算资源上运行；每个 DAgger 轮次配置/命令/作业文件先获本轮具体配方确认，规则见 `AGENTS.md`。Slurm、下载和 GPU 状态使用前重新观测。
- checkpoint 通过完整性检查后仍需模型精度、Thor 本机推理和 Thor↔3588 直连 smoke；真机测试还需现场安全条件与用户反馈复核。见 [05](../../05_inference_and_rollout.md)、[08](../../08_thor_edge_deployment.md)。
- 当前事实写入唯一 owner；kernel 仅在跨主题恢复确有必要时替换摘要，失败与原因按需写入 [07](../../07_change_log.md)。
