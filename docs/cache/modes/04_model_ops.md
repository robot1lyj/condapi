# 04 · Mode — Model Operations

用于 YAM LeRobot 数据、Pi0.5 LoRA、乐高分拣和后续 DAgger。

## 入口

- 数据格式、14D 动作和 norm stats → `docs/04_data_contracts.md`
- 训练、评估和 checkpoint gate → `docs/03_training_and_evaluation.md`
- 服务与安全 rollout → `docs/05_inference_and_rollout.md`
- Thor 端侧转换与本地验收 → `docs/08_thor_edge_deployment.md`
- 环境与服务器 → `docs/02_installation_and_environment.md`

## 操作边界

- 先审计数据集版本、任务 prompt、三路图像、14D state/action 和 norm stats，再启动训练。
- 当前第一阶段是乐高分拣监督微调；第二阶段收集并清洗人工纠正数据后再做 DAgger，不能混用数据集和 checkpoint。
- 使用 `pi05_yam_lora` 或其任务配置；模型内部可为 32D，YAM 输出必须回到真实 14D。
- 长任务使用 tmux；记录节点、GPU、实际命令、checkpoint 和失败原因。
- 真机 rollout 先做 Thor 本地策略 transform/golden comparison；只有控制器异机时才做 WebSocket smoke，再进入低速、限位和人工急停可用的测试。

## 结果写回

稳定默认写入 `kernel.md`；当前训练/数据事实写入 `docs/03` 或 `docs/04`；历史原因写入 `docs/07_change_log.md`。不要恢复 OpenArm/KAI0 研究计划作为当前入口。
