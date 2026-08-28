# OpenPI + OpenArm 后训练与 Rollout

本仓库是 OpenPI 的 OpenArm 定制分支，主线任务是双臂 OpenArm 的 T-shirt folding VLA：监督微调、KAI0/AWBC、Evo-RL、HIL 和真机 rollout。`examples/`、`third_party/` 与 Piper 仅作 legacy/reference，不是默认入口。

## 从这里开始

1. [交接索引](docs/00_handoff_index.md)：文档路由和状态词。
2. [系统架构](docs/01_system_architecture.md)：代码、数据流和边界。
3. [服务器与环境](docs/02_installation_and_environment.md)：服务器、conda、数据预检和训练命令。
4. [训练与评估](docs/03_training_and_evaluation.md)：配置、训练、checkpoint 和离线评估。
5. [数据合同](docs/04_data_contracts.md)：数据格式、单位、split 和 norm stats。
6. [推理与 rollout](docs/05_inference_and_rollout.md)：服务、初始位姿、WebSocket、HIL 和 RTC。
7. [研究计划](docs/06_openarm_research_plan.md)：KAI0、Evo-RL、Hybrid 的当前状态。

变更原因和结果只看 [07 · 变更历史](docs/07_change_log.md)。Piper 只看 [legacy 指引](docs/reference/legacy/piper.md)。

## 固定合同

```text
输入：base、left wrist、right wrist 三路图像 + OpenArm state + prompt
任务：Fold the T-shirt properly
state/action：16D = [右臂7关节, 右夹爪, 左臂7关节, 左夹爪]
训练单位：关节 degree；HQ 夹爪 motor degree，0=open，-66=closed
输出：50 步 action chunk；模型内部 32D，机器人输出 16D
```

OpenArm 只使用 `LeRobotOpenArmDataConfig`、`OpenArmInputs` 和 `OpenArmOutputs`。弧度、归一化夹爪和硬件限幅只在机器人客户端/ROS 边界处理；不能接入 Piper 14D 或旧单位数据。

## 最短可用路径

### 1. 安装

联网机器构建离线包；将仓库和 `artifacts/pi-conda-offline-bundle` 同步到目标机后，在目标机安装：

```bash
# 联网机器
git submodule update --init --recursive
bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle
```

```bash
# 目标机，当前目录为同步后的仓库
bash scripts/conda/install_offline_bundle.sh \
  --bundle-dir artifacts/pi-conda-offline-bundle \
  --openpi-dir "$PWD" --env-name pi-conda
```

### 2. 训练前检查

```bash
conda run -n pi-conda python -m pytest scripts/train_test.py -q
```

服务器数据、norm、正式 K-Policy 多节点训练和评估命令分别见 [02](docs/02_installation_and_environment.md)、[03](docs/03_training_and_evaluation.md) 和 [04](docs/04_data_contracts.md)。

### 3. 服务与 smoke

```bash
CHECKPOINT_DIR=/share/home/linyongjia/output/openpi/CONFIG/EXP_NAME/STEP
conda run -n pi-conda python scripts/serve_policy.py \
  --port 6666 \
  --force-prompt 'Fold the T-shirt properly, Advantage: positive' \
  --rtc-mode off \
  policy:checkpoint \
  --policy.config=pi05_openarm_kai0_awbc_v1 \
  --policy.dir="$CHECKPOINT_DIR"
```

端口监听不是部署成功。必须运行 [真实 WebSocket smoke](docs/05_inference_and_rollout.md#4-真实-websocket-smoke)，确认 `(50,16)`、单位、夹爪语义和 checkpoint 一致。

## 初始位姿与可调参数

- OpenArm 初始/复位位姿不在本仓库的 policy server/config 中；主线机器人仓库是 `/home/lyj/openarm_ros2_docker`。真机推理改 `scripts/start_real_inference_openpi.sh` 与 `scripts/start_real_inference_lerobot.sh` 的 `home_all()`，HIL 改 `scripts/start_real_hil_dagger_openpi.sh`，完整说明见该仓库 `docs/02_parameters_and_home.md`。
- 一次性真机回零在 bringup 后执行：`ros2 run openarm_arm openarm-arm home both --position 0 0 0 0 0 0 0 --gripper 0.9 --duration-sec 5 --rate-hz 50 --wait-for-command-slot-sec 2`。7 个关节值是 ROS 弧度，`0.9` 是 ROS 夹爪开口量；训练侧仍是 degree/HQ `0/-66`。先低速验证限位、碰撞和急停，再确认相机/数据分布；不要把位姿写进 `--rtc-metadata`。
- `examples/aloha_real/constants.py` 的 `START_ARM_POSE` 和 `examples/aloha_real/real_env.py` 的 `DEFAULT_RESET_POSITION` 只属于 ALOHA legacy，不能复制给 OpenArm。
- 训练参数改 `src/openpi/training/config.py` 或通过训练 CLI 覆盖；服务器路径、GPU、hosts、batch、workers 和 coordinator 看 [02](docs/02_installation_and_environment.md#可调参数)。服务参数和 action-chunk 执行策略看 [05](docs/05_inference_and_rollout.md#2-参数与初始位姿)。

## 交接检查

- [ ] 已按 `AGENTS.md` → `docs/cache/kernel.md` → `docs/cache/context_index.md` → 一个 mode pack 启动。
- [ ] 已确认远端 commit、数据目录、episode split、norm stats 和 checkpoint 是同一版本。
- [ ] 已通过真实 WebSocket smoke；真机 rollout 使用 tmux、急停、人工接管和可回退 checkpoint。
- [ ] 新事实只写入一个 owner 文档；历史结果才写入 [07](docs/07_change_log.md)。

Python 3.11、conda 离线依赖、Ruff/pytest 和提交边界见 [AGENTS.md](AGENTS.md)。本分支不提交凭据、不上传权重、不执行 `git push`。
