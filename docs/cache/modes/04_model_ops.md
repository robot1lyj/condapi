# 04 · Mode — Model Operations

用于 OpenArm 数据、监督微调、KAI0/AWBC、Evo-RL、HIL 采集和 rollout 评估。

## 先确认

- 任务和 16D/degree/HQ 夹爪合同 → `docs/04_data_contracts.md`
- 配置、norm、训练、评估 gate → `docs/03_training_and_evaluation.md`
- 服务、prompt、WebSocket、HIL/RTC rollout → `docs/05_inference_and_rollout.md`
- 当前 KAI0/Evo-RL/Hybrid 结论和待办 → `docs/06_openarm_research_plan.md`

## 实验纪律

- 区分 implemented、planned、historical；计划中的 HIL 或 E-Value 不能写成已完成。
- 新数据必须有独立版本、元数据和审计结果；不要覆盖原始数据或复用不同单位合同的 norm stats。
- SFT、KAI0 和 Evo-RL 保持独立实验名、初始化 checkpoint、数据集和评价协议，才能归因。
- 低成功率策略优先采集失败前缀、接管动作和恢复结尾；不能只增加普通成功示范后宣称阶段问题已解决。

## 结果写回

固定协议下的一组实验只在结束后写一条 `docs/07_change_log.md` 记录；新的稳定阈值、路径或配置写回其唯一 owner，不复制到本 mode。
