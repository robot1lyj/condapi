# Context Index

Router only. No facts, commands, metrics, or architecture details.

## Budget
- <=100 lines.
- New mode pack must fill a clear owner gap.
- Update existing modes before adding new ones.

## Mode Packs
- `docs/cache/modes/code_change.md` — openpi 代码修改、训练配置、数据 pipeline、验证与提交
- `docs/cache/modes/deployment.md` — 训练服务器部署、离线包传输、conda 环境、远端执行

## Canonical Docs
- Product baseline: `README.md`
- Documentation index: `docs/README.md`
- Current OpenArm RECAP plan: `docs/openarm_recap_reproduction_plan.md`
- Change history: `docs/CHANGELOG.md`
- Dataset versioning: `docs/dataset_versioning.md`
- Remote inference: `docs/remote_inference.md`
- Norm stats: `docs/norm_stats.md`
- Piper historical/training docs: `docs/piper_conda_training.md`, `docs/piper_dual_arm_training.md`
- Decisions: `docs/decisions/README.md`
- Long references: `docs/reference/00_reference_index.md`

## Quick Route
- 代码修改 / 配置 / 训练 / 测试 / 提交 -> `modes/code_change.md`
- 服务器部署 / 离线包 / conda 环境 / 远端训练 -> `modes/deployment.md`
- OpenArm RECAP 复现 / HIL / Evo-RL / KAI0 / AWBC -> `docs/openarm_recap_reproduction_plan.md`
- 远端推理服务 -> `docs/remote_inference.md`
- Piper 双臂历史流程 -> `docs/piper_conda_training.md`
