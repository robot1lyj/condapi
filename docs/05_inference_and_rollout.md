# 05 · 训练后 policy 端侧与兼容 smoke

本页保留 YAM 训练后 policy 的输入输出协议、Thor 本地推理验收和远程 WebSocket 兼容路径。默认运行位置是 NVIDIA Jetson AGX Thor；YAM 机械臂控制、CAN、GUI、home pose、控制频率和真机安全不在本仓库适配范围，也不要从独立 YAM-ABC-Reproduce 代码推断本项目合同。

Thor 系统、容器、Pi0.5 转换和 TensorRT 方案见 [08 · Thor 端侧部署](08_thor_edge_deployment.md)。

## 1. 部署前 gate

必须同时满足：

1. checkpoint 参数元数据完整，且有与训练数据绑定的 `assets/yam/norm_stats.json`。
2. 配置是 `pi05_yam_lora`（或明确记录的 YAM 配置），数据和 checkpoint 属于同一 14D 合同。
3. Thor 已核验 JetPack/L4T、GPU、Docker runtime 和容器内 CUDA/PyTorch/TensorRT；具体系统基线见 [08](08_thor_edge_deployment.md)。
4. JAX checkpoint 已作为 golden reference 保存；PyTorch 和 TensorRT/其他加速后端均需使用相同输入做数值比较。
5. 真正的本地推理 smoke 确认输入键、输出 shape、有限值和 checkpoint/norm 绑定；只有控制器不在 Thor 时才额外做远程 WebSocket smoke。

## 2. Thor 本地推理边界

默认数据流为：

```text
Thor camera/state/prompt
  -> YamInputs + norm
  -> PyTorch BF16 或 TensorRT engine
  -> YamOutputs + absolute action
  -> 有限的 (50,14) YAM action chunk
```

端侧 bundle 必须来自当前 `pi05_yam_lora` checkpoint。成熟案例的 `pi05_libero` 是 7D、horizon 10；不能直接复用其权重资产、TensorRT engine 或 49/54 ms benchmark 来代表 YAM。YAM 需要三路图像、模型内部 horizon 50 和真实 14D 输出，详见 [数据合同](04_data_contracts.md)。

首个端侧运行顺序固定为：

1. JAX reference 以实际 YAM 样本生成 golden 输出。
2. Thor 容器内转换为 PyTorch SafeTensors，先跑 BF16 PyTorch smoke。
3. 按 horizon 50 导出并构建 TensorRT FP8；通过真实样本误差 gate 后再尝试 NVFP4。
4. 在 Thor 本地直接调用 policy，验证三路图像、14D state、prompt 和 `(50,14)` 输出。

不要把直接 JAX runtime 作为首次默认路径；JAX 保留为转换源和数值基线。若后续使用 FlashRT 或 `openpi-thor`，也必须通过同一 YAM golden gate。

## 3. YAM 输入输出合同

YAM policy 输入使用和 LeRobot 导出一致的键：

```text
observation.state                  float array, shape (14,)
observation.images.top_rgb         RGB image
observation.images.left_rgb        RGB image
observation.images.right_rgb       RGB image
prompt                              scalar string（或由 task 注入）
```

图像可由 `YamInputs` 兼容 CHW/HWC，正式 smoke 建议使用 HWC `uint8`。推理返回：

```text
actions: float array, shape (50, 14), all finite
```

模型内部 padding 到 32D 只属于 OpenPI 模型边界，不应把 32D 直接当作 YAM 真实动作发送给机器人侧。单位、限幅、执行频率和安全检查由已核实的机器人侧系统负责。

## 4. 本地 smoke 验收

不得以容器启动、端口监听或只返回 metadata 作为成功。按顺序检查：

- 同一批三路实际图像、14D state 和 prompt 在 JAX reference、PyTorch、TensorRT（或 FlashRT）中都能完成推理；
- 输出严格为 `(50,14)`，所有值有限，action 顺序和 norm asset 绑定到 YAM；
- 记录 engine/backend、config、checkpoint、训练 commit、norm 路径、JetPack/L4T、CUDA/TensorRT、warmup、时延和功耗模式；
- 端侧本地路径通过后，才允许进入低速、限位和人工急停可用的机器人侧测试。

当前仓库尚未提供独立的 YAM Thor 本地 smoke 脚本；先用 `yam_policy_test.py` 验证 14D transform，再在 Thor 容器内补充真实样本和 golden comparison。不要把旧 OpenArm 16D 工具重新作为默认入口。

## 5. 远程 WebSocket 兼容路径

当机器人控制器或相机进程暂时在另一台设备时，可以保留通用 WebSocket 服务；这不是后续默认推理架构。服务端仍需显式指定 checkpoint 和配置：

```bash
export REPO_ROOT=/home/wuyan/lyj/YAM/YAM_code
export CHECKPOINT_DIR=/path/to/complete/yam_pi05_lora_checkpoint
export PORT=8000

cd "$REPO_ROOT"
tmux new-session -s yam-policy -c "$REPO_ROOT"
```

在 tmux 中执行：

```bash
"$PYTHON" scripts/serve_policy.py \
  --port "$PORT" \
  --rtc-mode off \
  policy:checkpoint \
  --policy.config=pi05_yam_lora \
  --policy.dir="$CHECKPOINT_DIR"
```

`--policy.repo-id` 不是当前 CLI 参数；如服务构造 transform 需要覆盖数据 repo，使用已有的 `--policy-repo-id`，但不能替代 checkpoint 内 norm stats。RTC 首轮保持 `off`，任何 RTC 改动都必须保留普通推理路径并能自动回退。

使用真实 `openpi-client` WebSocket 协议发送三路固定图像、14D state 和 prompt，检查：

- 输入无缺少 image/state/prompt 错误；
- 输出严格为 `(50,14)`，无 NaN/Inf；
- metadata 的 `robot_action_dim/output_action_dim=14`、`model_action_dim/action_dim=32`、`action_horizon=50` 与 checkpoint config 一致；
- 日志记录 commit、checkpoint、config、端口、prompt 和 smoke 结果。

服务停止前保存日志和 smoke 报告，在对应 tmux 中 `Ctrl-c`；不要 kill 其他用户进程或现有下载/训练作业。

## 6. 停止条件

本地或远程路径出现以下任一情况，停止 rollout 并回到 checkpoint/data gate：

- shape 不是 14D 输入或 50x14 输出；
- 动作非有限、checkpoint/norm 不匹配、转换产物未通过 JAX reference 比较；
- 三路图像缺失、通道/时间同步不明；
- 把 OpenArm/Piper 单位、顺序或 transform 混入 YAM；
- 只看到端口监听或服务进程存在，尚无真实推理结果。
