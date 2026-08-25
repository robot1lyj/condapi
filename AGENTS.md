# Repository Guidelines

## 项目定位

本仓库是 OpenPI 的 OpenArm 定制分支，默认任务是双臂 T-shirt folding 的 VLA 后训练与真机 rollout。Piper 配置和上游示例仍可运行，但只属于 legacy/reference，不得作为 OpenArm 默认路径或新实验结论。

## Context OS 记忆规则

- 唯一记忆系统是 `AGENTS.md` + `docs/cache/`；不要新增 `.agents`、`.codex` 或其他平行缓存。
- 启动时依次读取本文件、`docs/cache/kernel.md`、`docs/cache/context_index.md`，然后按路由最多读取一个 `docs/cache/modes/*.md`。
- 稳定事实只保留一个 owner：热默认写 kernel，任务路由写 index，操作边界写 mode，详细事实写编号化 `docs/`，历史原因写 `docs/07_change_log.md`。
- 记忆预算：`kernel.md` 不超过 80 行，`context_index.md` 不超过 100 行，每个 mode 不超过 80 行。
- 新增事实先合并/压缩旧 owner，再写入；不要把命令、指标和架构复制到 index 或 kernel。

## 编号化文档所有权

- `docs/00_handoff_index.md`：交接导航和当前/计划/历史边界。
- `docs/01_system_architecture.md`：代码与运行架构。
- `docs/02_installation_and_environment.md`：环境、路径和远端资源。
- `docs/03_training_and_evaluation.md`：训练、评估和 checkpoint gate。
- `docs/04_data_contracts.md`：数据格式、单位、动作维度和 norm stats。
- `docs/05_inference_and_rollout.md`：服务、WebSocket、HIL、RTC 和安全 rollout。
- `docs/06_openarm_research_plan.md`：KAI0/Evo-RL/Hybrid 当前研究计划；不得在其他文档复制整段实验计划。
- `docs/07_change_log.md`：按日期记录原因和结果，不替代当前操作说明。
- `docs/reference/`：长篇技术参考或 legacy；默认入口不依赖其中的旧结论。

## 代码与目录

- `src/openpi/`：模型、策略、训练、数据 transform 和公共工具。
- `packages/openpi-client/`：机器人侧 WebSocket/IO 客户端。
- `scripts/`：训练、数据准备、服务和审计入口。
- `examples/`、`third_party/`：平台示例或 vendored 依赖，除非任务明确要求，不改其 README 来表达本项目事实。

## 环境与常用命令

Python 3.11，行宽 120，GPU 是训练/大多数推理的前提。依赖使用 conda 离线包，不使用 uv：

```bash
git submodule update --init --recursive
bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle
bash scripts/conda/install_offline_bundle.sh --bundle-dir artifacts/pi-conda-offline-bundle --env-name pi-conda
conda run -n pi-conda pip install --no-build-isolation --no-deps -e .
conda run -n pi-conda pip install --no-build-isolation --no-deps -e packages/openpi-client
```

验证和格式化：

```bash
ruff check .
ruff format .
conda run -n pi-conda python -m pytest --strict-markers -m "not manual"
```

训练/服务的项目化入口见 `docs/03_training_and_evaluation.md`、`docs/05_inference_and_rollout.md`，不要在本文件堆积实验参数。

## 不可违反的边界

- OpenArm 合同固定为 16D `[右臂7, 右夹爪, 左臂7, 左夹爪]`；训练为 degree/HQ 夹爪 `0=open,-66=closed`，ROS 弧度/归一化只在客户端边界转换。
- OpenArm 只能使用 `LeRobotOpenArmDataConfig`、`OpenArmInputs`、`OpenArmOutputs`；禁止接入 Piper 14D transform。
- 不提交凭据、token、私钥、服务器密码；不删除远端数据/权重/缓存，除非用户明确授权。
- 长训练、服务和真机操作使用 tmux；端口监听不等于推理可用，必须做真实 WebSocket smoke。
- 任何 RTC 改动都必须保留旧推理路径，并可通过 `rtc_mode` 关闭或自动回退。

## Git 自动化

完成请求后运行 `git add -A` 和中文 commit，例如 `git commit -m "重构项目交接文档"`；永远不要 `git push`。提交前至少执行 `git diff --check`，代码改动再执行 Ruff/pytest。
