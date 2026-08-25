# OpenPI + OpenArm 后训练与 Rollout 工程

这是 OpenPI 的 OpenArm 定制分支，当前工作重点是双臂 OpenArm 上的 T-shirt folding VLA：监督微调、KAI0/AWBC、Evo-RL、HIL 和真机 rollout。上游 OpenPI 的通用示例仍保留在 `examples/` 与 `third_party/`，但不再作为本项目的交接入口；Piper 文档只用于历史追溯。

## 先读什么

接手者按下面顺序阅读，避免把历史实验当成当前默认路径：

1. [交接索引](docs/00_handoff_index.md)：当前/计划中/历史内容的唯一导航。
2. [系统架构](docs/01_system_architecture.md)：代码、数据、训练和服务边界。
3. [环境与安装](docs/02_installation_and_environment.md)：本地和远端 conda、路径及安全要求。
4. [训练与评估](docs/03_training_and_evaluation.md)：通用训练入口、OpenArm 正式配置和验收。
5. [数据合同](docs/04_data_contracts.md)：16D 动作、单位、LeRobot 数据和 norm stats。
6. [推理与 rollout](docs/05_inference_and_rollout.md)：服务端、WebSocket smoke、HIL 和 RTC。
7. [研究计划](docs/06_openarm_research_plan.md)：KAI0、Evo-RL、Hybrid 的当前实验状态与下一步。

历史变更只看 [CHANGELOG](docs/07_change_log.md)；RTC 实现边界和 Piper legacy 指引位于 [reference](docs/reference/00_reference_index.md)，不应覆盖当前文档。

## 当前项目边界

```text
输入：三路相机 + OpenArm state + prompt
模型：π0/π0.5（JAX 为主，pi0/pi0.5 支持 PyTorch）
输出：50 步动作块，OpenArm 16D
任务：Fold the T-shirt properly
```

OpenArm 的 16D 顺序固定为 `[右臂7关节, 右夹爪, 左臂7关节, 左夹爪]`。训练数据中的机械臂关节使用 degree，HQ 夹爪使用电机角度 `0=open`、`-66=closed`；ROS/运行时的弧度和归一化转换只允许发生在客户端边界。OpenArm 必须使用 `LeRobotOpenArmDataConfig`、`OpenArmInputs` 和 `OpenArmOutputs`，不能套用 Piper 的 14D transform。

## 已实现与当前状态

- OpenArm HQ/Site 数据清洗、Stage scorer、KAI0 二值 AWBC 数据构建和正式 `pi05_openarm_kai0_awbc_v1` 配置已在仓库中。
- 80k K-Policy、16 个 checkpoint 的双域 sweep 和部署 smoke 已记录完成；离线规则选择 20k，但现有真机 A/B 未优于 79999，交接记录中的当前 collector 是 79999。
- 下一批是 HIL-T30：错误对角线恢复、重复甩平恢复、已展开但未进入折叠各 10 条。它是计划，不等同于“已经采集完成”；先检查 `docs/06_openarm_research_plan.md` 和远端状态。
- Evo value/infer、正式 ACP 训练和 KAI0+Evo Hybrid 仍属于后续路线；配置中 ACP dropout 的正式目标为 `0.3`，不能把现有 probe 的 `0.0` 当最终结论。

## 最短可用路径

### 本地准备

```bash
git submodule update --init --recursive
bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle
bash scripts/conda/install_offline_bundle.sh \
  --bundle-dir artifacts/pi-conda-offline-bundle --env-name pi-conda
conda run -n pi-conda pip install --no-build-isolation --no-deps -e .
conda run -n pi-conda pip install --no-build-isolation --no-deps -e packages/openpi-client
```

### 训练前检查

```bash
conda run -n pi-conda python scripts/train_test.py
```

norm stats、正式 OpenArm 多节点命令、数据集参数和 smoke gate 见 [训练与评估](docs/03_training_and_evaluation.md) 与 [数据合同](docs/04_data_contracts.md)，不要从这里复制一个未经核对的旧 checkpoint 路径。

### 服务与 rollout

```bash
conda run -n pi-conda python scripts/serve_policy.py \
  --port 6666 \
  --force-prompt 'Fold the T-shirt properly, Advantage: positive' \
  policy:checkpoint \
  --policy.config=pi05_openarm_kai0_awbc_v1 \
  --policy.dir=<CHECKPOINT_DIR>
```

端口监听不算部署成功。必须从真实 WebSocket 客户端请求一次，确认有限的 `(50, 16)` 动作、50 步 horizon、degree/HQ 夹爪元数据和强制 prompt；完整流程见 [推理与 rollout](docs/05_inference_and_rollout.md)。

## 交接时必须确认

- [ ] 已读 `AGENTS.md`、`docs/cache/kernel.md`、`docs/cache/context_index.md`，再按任务只读一个 mode pack。
- [ ] 已确认远端仓库、conda 环境、数据集和输出目录，不把本地路径误当远端路径。
- [ ] 已确认 checkpoint 是完整 Orbax 保存，而不是只有数字目录或半写入的 `params/`。
- [ ] 已完成数据 `meta/info.json`、视频尾帧、norm stats、16D/单位和 prompt 检查。
- [ ] 真机前使用 tmux、低风险动作和可回退 checkpoint；任何 RTC 改动都保留 `rtc_mode=off` 旧路径。
- [ ] 新事实只写入一个 owner 文档，并在 [CHANGELOG](docs/07_change_log.md) 留下简短原因和结果。

## 开发约定

Python 3.11，行宽 120；常用检查为 `ruff check .`、`ruff format .` 和 `conda run -n pi-conda python -m pytest --strict-markers -m "not manual"`。详细规则、提交要求和 Context OS 记忆边界见 [AGENTS.md](AGENTS.md)。

本分支不自动上传权重、不提交凭据、不执行 `git push`。接手者完成改动后应提交中文 commit，推送由项目负责人执行。
