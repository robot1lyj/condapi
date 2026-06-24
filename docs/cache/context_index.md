# Context Index

Router only — no facts, commands, metrics, or architecture details.

## Budget
- ≤100 lines.
- New mode pack must fill a clear owner gap.
- Update existing modes before adding new ones.

## Mode Packs
- `docs/cache/modes/code_change.md` — openpi 代码修改、训练配置、数据 pipeline、验证与提交
- `docs/cache/modes/deployment.md` — 训练服务器部署、离线包传输、conda 环境、远端执行

## Canonical Docs
- Product baseline: `README.md`
- Architecture: `docs/ARCHITECTURE.md`
- Change history: `docs/CHANGELOG.md`
- Piper conda training: `docs/piper_conda_training.md`
- Remote inference: `docs/remote_inference.md`
- Norm stats: `docs/norm_stats.md`

## Quick Route
- 代码修改 / 配置 / 训练 / 测试 / 提交 → `modes/code_change.md`
- 服务器部署 / 离线包 / conda 环境 / 远端训练 → `modes/deployment.md`
- 远端推理服务 → `docs/remote_inference.md`
- Piper 双臂完整训练流程 → `docs/piper_conda_training.md`
