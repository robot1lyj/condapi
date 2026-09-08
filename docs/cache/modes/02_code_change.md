# 02 · Mode — Code Change

用于修改 `src/`、`scripts/`、`packages/`、配置或当前文档，当前默认目标是 YAM 双臂训练。

## 工作顺序

1. 先看 `git status --short`、最近提交和相关源文件，确认没有覆盖用户改动。
2. 按授权进行改造，4 空格、Ruff 行宽 120；控制层支持 Python 3.11+，Pi 模型环境仍按独立依赖要求。Pi 的 YAM 配置放 `src/openpi/training/config.py`；其他模型优先添加声明和原生配置引用，共享 `adapters/lerobot/` 入口，不复制 trainer/processor。
3. 代码改动运行 `ruff check .`、`ruff format .` 和非 manual pytest；文档改动至少运行 `git diff --check` 与链接/旧路径审计。
4. 影响默认行为、数据合同或部署边界时，同步更新唯一 owner 文档和 `docs/07_change_log.md`。
5. 核对仅包含本任务内容后提交，推送当前功能分支到 Gitea `origin` 和 GitHub `github`；改造工作树不自动合并 main，不部署远端，不 force push。

## 代码边界

- Pi 的 YAM 路径使用 `LeRobotYamDataConfig`、`YamInputs/Outputs`；其他系列用各自原生 processor，共享双臂 14D `[左6+夹爪, 右6+夹爪]` 的机器人语义合同。
- 不把 YAM transform 与 OpenArm 16D 或 Piper transform 混用。
- RTC 改动必须保留旧推理路径，并可由 `rtc_mode` 关闭或回退。
- 不新增不必要依赖；先检查 `requirements-pi-pip.txt` 和现有 conda 环境。

## 文档写回

- 架构事实 → `docs/01_system_architecture.md`
- 安装/训练/服务操作 → `docs/02`、`docs/03`、`docs/05`
- 数据合同 → `docs/04_data_contracts.md`
- 新框架操作/状态与模型接入 → `docs/10_vla_platform.md`；不要把当前研究写进历史 OpenArm 归档。
- 原因、事故、结果 → `docs/07_change_log.md`
