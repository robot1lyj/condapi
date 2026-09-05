# 05 · 训练后 policy 端侧与 IPC smoke

本页保留 YAM 训练后 policy 的输入输出协议、Thor 本地推理验收和 Thor↔3588 的直连以太网推理通道。模型运行在 NVIDIA Jetson AGX Thor；3588 负责相机采集和机械臂控制，YAM 机械臂控制、CAN、GUI、home pose、控制频率和真机安全不在本仓库适配范围，也不要从独立 YAM-ABC-Reproduce 代码推断本项目合同。

Thor 系统、容器、Pi0.5 转换和 TensorRT 方案见 [08 · Thor 端侧部署](08_thor_edge_deployment.md)。

## 1. 部署前 gate

必须同时满足：

1. checkpoint 参数元数据完整，且有与训练数据绑定的 `assets/yam/norm_stats.json`。
2. 配置是 `pi05_yam_lora`（或明确记录的 YAM 配置），数据和 checkpoint 属于同一 14D 合同。
3. Thor 已核验 JetPack/L4T、GPU、Docker runtime 和容器内 CUDA/PyTorch/TensorRT；具体系统基线见 [08](08_thor_edge_deployment.md)。
4. 原 JAX checkpoint 与 LoRA、golden 输入/噪声/输出已保存；任何候选后端均需按 [08](08_thor_edge_deployment.md) 做分阶段精度验收，不能只与转换后的 Torch 比较。
5. Thor 本地推理 smoke 确认输入键、输出 shape、有限值和 checkpoint/norm 绑定；随后必须做 3588↔Thor 的真实直连以太网 smoke，验证 observation/action 往返。

## 2. Thor 本地推理边界

默认数据流为：

```text
3588 camera/state/prompt
  == direct Ethernet / WebSocket or agreed transport ==>
Thor YamInputs + norm
  -> 已通过原 JAX 精度验收的本地 policy（runtime 待实测确定）
  -> YamOutputs + absolute action
  == direct Ethernet / action response ==>
3588 controller: 有限的 (50,14) YAM action chunk
```

端侧 bundle 必须来自当前 `pi05_yam_lora` checkpoint。成熟案例的 `pi05_libero` 是 7D、horizon 10；不能直接复用其权重资产、TensorRT engine 或 49/54 ms benchmark 来代表 YAM。YAM 需要三路图像、模型内部 horizon 50 和真实 14D 输出，详见 [数据合同](04_data_contracts.md)。

首个端侧运行顺序固定为：

1. 原 JAX policy 以实际 YAM 样本和同一份噪声数组生成 golden；保留原始 LoRA checkpoint。
2. 核验 Thor 原生 JAX 可行性；若转换到 Torch，先审计 LoRA 合并、FP32 构造/存储、norm 绑定和未量化计算对齐。
3. 精度通过后再按延迟需求决定是否导出 engine；FP8/NVFP4 和定制 FP16 是独立候选，必须与原 JAX 比较。
4. 在 Thor 本地用回放样本直接调用 policy，验证三路图像、14D state、prompt 和 `(50,14)` 输出；再用 3588 的真实 observation 做跨 IPC 直连 smoke。

原生 JAX、经审计的 Torch 和加速 engine 均须有 Thor 实测证据；具体精度、环境限制和验收层次由 [08](08_thor_edge_deployment.md) 持有。保持 `action_horizon=50` 与默认去噪 `num_steps=10` 分别记录。跨 IPC smoke 只证明传输和模型可运行，不能替代任务成功率验收。

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

## 5. Thor↔3588 网络推理通道

Thor 服务端只在 Thor 上加载 checkpoint，3588 作为直连以太网客户端发送 observation 并接收 action；这不是远程模型推理，而是两台 IPC 的生产数据通道。服务端仍需显式指定 checkpoint 和配置：

```bash
export THOR_REPO_ROOT=/path/to/condapi-on-thor
export CHECKPOINT_DIR=/path/to/complete/yam_pi05_lora_checkpoint
export PORT=8000

cd "$THOR_REPO_ROOT"
tmux new-session -s yam-policy -c "$THOR_REPO_ROOT"
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

使用真实 `openpi-client` WebSocket 协议（或后续核定的等价直连协议）从 3588 发送三路图像、14D state 和 prompt，检查：

- 输入无缺少 image/state/prompt 错误；
- 输出严格为 `(50,14)`，无 NaN/Inf；
- metadata 的 `robot_action_dim/output_action_dim=14`、`model_action_dim/action_dim=32`、`action_horizon=50` 与 checkpoint config 一致；
- 日志记录 commit、checkpoint、config、端口、prompt 和 smoke 结果。

服务停止前保存 Thor 日志和跨 IPC 直连 smoke 报告，在 Thor 对应 tmux 中 `Ctrl-c`；不要触碰 3588 的系统、控制进程、相机进程或其他用户任务。

## 6. 停止条件

Thor 本地或跨 IPC 直连路径出现以下任一情况，停止 rollout 并回到 checkpoint/data gate：

- shape 不是 14D 输入或 50x14 输出；
- 动作非有限、checkpoint/norm 不匹配、转换产物未通过 JAX reference 比较；
- 三路图像缺失、通道/时间同步不明；
- 把 OpenArm/Piper 单位、顺序或 transform 混入 YAM；
- 只看到端口监听或服务进程存在，尚无 Thor 本地推理和 3588↔Thor 网络结果。
