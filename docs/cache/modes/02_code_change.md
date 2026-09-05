# 02 · Mode — Code Change

用于修改 `src/`、`scripts/`、`packages/`、配置或当前文档，当前默认目标是 YAM 双臂训练。

## 工作顺序

1. 先看 `git status --short`、最近提交和相关源文件，确认没有覆盖用户改动。
2. 小范围修改，保持 Python 3.12、4 空格、Ruff 行宽 120；新增 YAM 配置放 `src/openpi/training/config.py`。
3. 代码改动运行 `ruff check .`、`ruff format .` 和非 manual pytest；文档改动至少运行 `git diff --check` 与链接/旧路径审计。
4. 影响默认行为、数据合同或部署边界时，同步更新唯一 owner 文档和 `docs/07_change_log.md`。
5. 完成后 `git add -A`，用中文提交，并把同一提交推送到 Gitea `origin` 和 GitHub `github`；不 force push。

## 代码边界

- YAM 使用 `LeRobotYamDataConfig`、`YamInputs/Outputs` 和双臂 14D `[左6+夹爪, 右6+夹爪]` 合同。
- 不把 YAM transform 与 OpenArm 16D 或 Piper transform 混用。
- RTC 改动必须保留旧推理路径，并可由 `rtc_mode` 关闭或回退。
- 不新增不必要依赖；先检查 `requirements-pi-pip.txt` 和现有 conda 环境。

## 文档写回

- 架构事实 → `docs/01_system_architecture.md`
- 安装/训练/服务操作 → `docs/02`、`docs/03`、`docs/05`
- 数据合同 → `docs/04_data_contracts.md`
- 当前研究计划 → `docs/06_openarm_research_plan.md`
- 原因、事故、结果 → `docs/07_change_log.md`
