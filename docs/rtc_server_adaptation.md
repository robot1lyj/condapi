# OpenPI 服务器端 RTC 推理指南（pi05_piper_dual）

> 目标：在 `openpi` 服务端启用 Real-Time Chunking（RTC）推理，使远程推理在高延迟下仍连续、平滑。

## 快速开始（推荐）

灰度上线（兼容旧客户端）：
```bash
python scripts/serve_policy.py \
  --port 6666 \
  --rtc-mode=auto \
  --rtc-metadata docs/rtc_metadata_piper_dual.json \
  policy:checkpoint \
  --policy.config=pi05_piper_dual \
  --policy.dir /output/openpi/pi05_piper_dual/piper_ft_pi05/4999
```

强制 RTC（仅联调/验证）：
```bash
python scripts/serve_policy.py \
  --port 6666 \
  --rtc-mode=only \
  --rtc-metadata docs/rtc_metadata_piper_dual.json \
  policy:checkpoint \
  --policy.config=pi05_piper_dual \
  --policy.dir /output/openpi/pi05_piper_dual/piper_ft_pi05/4999
```

关闭 RTC（保持旧推理）：
```bash
python scripts/serve_policy.py \
  --port 6666 \
  --rtc-mode=off \
  policy:checkpoint \
  --policy.config=pi05_piper_dual \
  --policy.dir /output/openpi/pi05_piper_dual/piper_ft_pi05/4999
```

## 运行参数说明（服务端）

- `--rtc-mode`：`off`/`auto`/`only`
  - `off`：忽略 RTC，完全走旧推理路径。
  - `auto`：请求带 `rtc` 则走 RTC；失败或不支持自动回退旧推理（推荐默认）。
  - `only`：必须含 `rtc`，否则返回错误。
- `--rtc-metadata <path>`：RTC 握手元数据 JSON（仅服务端配置）。
  - 建议使用模板：`docs/rtc_metadata_piper_dual.json`。
  - `action_horizon/action_dim` 必须与模型一致（服务端会从模型补齐/校验）。
  - `control_hz/use_delta_joint_actions` 只透传给客户端；`use_delta_joint_actions` 表示模型是否在**delta 动作空间**训练。
- `--port`：服务端端口（默认 8000）。
- `--default-prompt`：可选固定 prompt。

## 兼容性说明

- **保持旧推理路径可用**：推荐 `--rtc-mode=auto`，满足 RTC 与旧客户端共存的要求。
- **仅 JAX pi0 路径支持 RTC**：如果 checkpoint 使用 PyTorch（含 `model.safetensors`），RTC 会回退普通推理或报错（`rtc_mode=only`）。

## Metadata 模板（服务端准备）

示例字段：
```
{
  "model_name": "pi05_piper_dual",
  "robot_type": "piper_shm",
  "repo_id": "/share/home/linyongjia/datasets/piper_pen_v001",
  "action_horizon": 50,
  "action_dim": 14,
  "robot_action_dim": 14,
  "control_hz": 30,
  "use_delta_joint_actions": true,
  "action_units": "absolute_radians",
  "input_keys": [
    "observation.state",
    "prompt",
    "observation.images.top_rgb",
    "observation.images.left_wrist",
    "observation.images.right_wrist"
  ],
  "image_keys": [
    "observation.images.top_rgb",
    "observation.images.left_wrist",
    "observation.images.right_wrist"
  ],
  "prompt_key": "prompt",
  "action_key": "action"
}
```

说明：
- `use_delta_joint_actions=true` 表示**模型内部**在 delta 空间训练；推理输出会经过 `AbsoluteActions` 转为绝对角。
- RTC 的 `prev_actions` 必须与模型内部空间一致（即 delta）；若客户端只有绝对角，需要先按 joint mask 做 `prev_actions_delta = prev_actions_abs - state`（夹爪保持绝对）。

## 0. 服务端注意事项（必看）
- **pi0.5/JAX 才能用 RTC**：目前 RTC 仅实现于 `pi0.py` 路径；若 checkpoint 目录含 `model.safetensors`（PyTorch），RTC 会回退普通推理或报错（`rtc_mode=only`）。
- **metadata 必须与模型一致**：`action_horizon/action_dim` 以服务端为准；若 payload 不一致会被拒绝或回退。
- **推荐 `rtc_mode=auto`**：保证旧客户端仍可用，RTC 失败自动回退，便于灰度。
- **metadata JSON 已提供模板**：`docs/rtc_metadata_piper_dual.json`，如有改动需同步更新。

## 1. 为什么要改（现状问题）
- 目前服务端是“同步式 chunk 推理”：客户端每次请求必须等待推理完成才能得到新动作块。
- 远程推理存在明显 RTT（120~160ms），会在 chunk 边界产生停顿或不连续跳变。
- 仅靠客户端 action chunking 无法消除“推理延迟导致的边界抖动”。RTC 的核心是**服务端根据上一段动作做 inpainting**，使新 chunk 与旧 chunk 在时间上连续。

## 2. 核心变量与约束
- `H`：预测 horizon（动作块长度，pi05_piper_dual 为 50）。
- `s`：执行 horizon（每次执行前 `s` 步）。
- `Δt`：控制周期（本地 30Hz -> 33.3ms）。
- `δ`：推理耗时（含网络）。
- `d`：推理延迟（以控制步数计）。

推理延迟定义：
```
 d := floor(δ / Δt)
```

关键约束：
```
 d <= s <= H - d
```

## 3. RTC 的关键公式（论文核心）

### (1) Flow matching 采样更新
```
A_t^{τ + 1/n} = A_t^τ + (1/n) * v_π(A_t^τ, o_t, τ)
```

### (2) Inpainting 引导（ΠGDM）
```
v_ΠGDM(A_t^τ, o_t, τ) =
  v(A_t^τ, o_t, τ)
  + min(β, (1 - τ) / (τ * r_τ^2)) * (Y - Â_t^1)^T * diag(W) * (∂Â_t^1 / ∂A_t^τ)
```

```
Â_t^1 = A_t^τ + (1 - τ) * v(A_t^τ, o_t, τ)
r_τ^2 = (1 - τ)^2 / (τ^2 + (1 - τ)^2)
```

### (5) Soft mask（跨 chunk 连续性）
```
W_i = { 1                                  if i < d
      { c_i * (e^{c_i} - 1) / (e - 1)      if d <= i < H - s
      { 0                                  if i >= H - s

c_i = (H - s - i) / (H - s - d + 1),  i in {0, ..., H - 1}
```

## 4. 服务端适配点（openpi）

### 4.1 WebSocket 协议扩展（必须后向兼容）
当前服务端直接把收到的 msgpack 解包为 `obs`，交给 `policy.infer(obs)`。
为兼容 RTC，需要支持新 payload：
```
{
  "type": "infer",
  "obs": {...},
  "rtc": {
    "prev_actions": <np.ndarray [H-s, action_dim]>,
    "d": <int>,
    "s": <int>,
    "action_horizon": <int>,
    "action_dim": <int>
  }
}
```

服务端逻辑：
- 若 payload 带 `obs` 字段，先取 `obs`；否则保持旧协议（payload 即 obs）。
- 若 payload 带 `rtc`，执行 RTC 推理；否则走现有推理。
- `rtc` 数据必须**在进入模型前剥离**，避免影响 `Observation.from_dict`。

修改位置建议：
- `openpi/src/openpi/serving/websocket_policy_server.py`

### 4.2 保留旧推理的兼容策略 + CLI 选项（建议）
- **默认行为不变**：不改 `policy.infer(obs)` 入口；旧客户端继续发送裸 `obs`，保持当前推理路径。
- **按需开启 RTC**：建议在 `scripts/serve_policy.py` 增加 `--rtc-mode {off,auto,only}`：
  - `off`（默认）：忽略 `rtc` 或返回明确错误，行为等同现在。
  - `auto`：payload 含 `rtc` 时走 RTC；否则走旧推理，便于灰度上线。
  - `only`：所有请求必须含 `rtc`，用于强制验证 RTC。
- **能力回报**：服务端 metadata/handshake 返回 `rtc_mode`、`action_horizon`、`control_hz`，便于客户端自适配。

示例：
```bash
python scripts/serve_policy.py \
  --port 6666 \
  --rtc-mode=auto \
  policy:checkpoint \
  --policy.config=pi05_piper_dual \
  --policy.dir /output/openpi/pi05_piper_dual/piper_ft_pi05/4999
```

### 4.3 Metadata/handshake 配置（推荐）
- **不要每次请求都附带 metadata**：增加 payload 且易版本不一致；握手阶段一次性发送即可。
- **服务端加载元数据**：建议使用模板文件并在启动时加载或注入到 `policy.metadata`。
- **客户端侧兜底**：metadata 缺失时可读本地 override 文件，仅用于本地对齐，不必上送。
- **CLI 指定**：`scripts/serve_policy.py` 支持 `--rtc-metadata <path>` 载入 JSON。
- **模板参考**：`docs/rtc_metadata_piper_dual.json`（建议与对应的版本化数据集目录保持一致）。

示例模板字段（节选）：
```
{
  "model_name": "pi05_piper_dual",
  "action_horizon": 50,
  "action_dim": 14,
  "robot_action_dim": 14,
  "control_hz": 30,
  "use_delta_joint_actions": true,
  "action_units": "absolute_radians",
  "input_keys": [
    "observation.state",
    "prompt",
    "observation.images.top_rgb",
    "observation.images.left_wrist",
    "observation.images.right_wrist"
  ],
  "image_keys": [
    "observation.images.top_rgb",
    "observation.images.left_wrist",
    "observation.images.right_wrist"
  ],
  "prompt_key": "prompt",
  "action_key": "action"
}
```

### 4.4 新增 RTC 推理入口
建议新增一个推理函数（示意命名）：
- `openpi/serving/rtc_infer.py` or `openpi/models/rtc_sampler.py`
- 提供 `sample_actions_rtc(observation, prev_actions, d, s, ...)`

关键逻辑：
1) 把 `prev_actions` 右侧 padding 到长度 `H`。
2) 计算 soft mask `W`（公式 5）。
3) 在采样循环中加入 ΠGDM 引导（公式 2/3/4）。
4) 返回完整 `A_new`（长度 `H`）。

### 4.5 修改采样过程（pi0 模型）
`pi0` 模型采样入口：
- `openpi/src/openpi/models/pi0.py :: sample_actions`

RTC 需要在采样循环中加入引导项：
- 可复制一份 `sample_actions_rtc`，或在 `sample_actions` 里检测 `rtc` 参数。
- `pi0` 里时间从 1 → 0（`dt = -1/num_steps`），公式中的 `τ` 可以直接视作当前 `time`。

JAX 计算向量-雅可比积建议：
- 使用 `jax.vjp` 或 `jax.grad` 计算 `∂Â_t^1/∂A_t^τ` 对应的 VJP。

## 5. 必要注意事项
- **动作语义**：`pi05_piper_dual` 在 `src/openpi/training/config.py` 中配置 `use_delta_joint_actions=True`，模型内部为 delta；推理输出经 `AbsoluteActions` 转为绝对角。RTC 的 `prev_actions` 必须是 delta，否则会出现“二次累加”。若客户端只持有绝对角，请用 `make_bool_mask(6, -1, 6, -1)` 对关节维做 `prev_actions_delta = prev_actions_abs - state`（夹爪保持绝对）。
- **d/s 计算**：`d` 必须是“步数”，而不是毫秒；建议取最近 `b` 次延迟的 `max`（更保守）。
- **s 约束**：若 `s < d`，RTC 不能保证连续；服务端可主动 clamp 为 `s = max(s, d)`。
- **β 裁剪**：小步数控制场景下必须做裁剪，否则引导项可能发散。
- **消息大小**：`prev_actions` 会增大 payload，建议依旧使用 msgpack（二进制）。

## 6. 推荐参数（对齐论文 + 我们场景）
论文真实系统：
- `H=50, Δt=20ms, n=5, s_min=25, β=5, b=10`

我们当前：
- `H=50, Δt=33.3ms(30Hz), RTT≈120~160ms -> d≈4~5`
- 推荐先设：`s_min=10~15, β=5, b=10`
- 若动作仍跳变：增大 `s_min` 或启用 RTC 引导。

## 7. 验证与排查清单
- `/healthz` 必须 OK。
- 服务端输出 metadata（至少 `action_horizon`），客户端可自动对齐。
- 先用 `dry_run_client.py --handshake-only` 验证 ws。
- 再用 `rtc_infer.py --dry-run` 验证 async chunk 是否有动作断档。
- 看延迟日志：`d_obs` 应稳定在 4~6。

## 8. 最小可用改造路径（建议顺序）
1) **先做协议扩展**（服务端能识别 `obs`/`rtc` envelope）。
2) **先跑“无 RTC 引导”的异步执行**，验证不卡顿。
3) **再接入 RTC inpainting**，解决 chunk 边界跳变。

## 9. 实施结论与疑问

### 9.1 已确认结论
- **d/s 计算归属**：`d` 由客户端基于端到端 RTT 估计并上报；服务端只使用该 `d` 做引导。推荐 `d = ceil(RTT / Δt)`，滑动窗口保守估计 `d_est = max(last_b)` 或 `p95 + 1`；最终确保 `s = max(d_est, s_min)` 且满足 `d <= s <= H - d`。RTT 以**推理请求的往返时间**为主（发包→收包），`/healthz` 仅连通性不适合统计。
- **prev_actions 来源**：推荐客户端每次携带，服务端尽量无状态；可加 `rtc.reset=true` 作为重置标志。若服务端维护状态，需要新连接清空 + `episode_id/reset_rtc` + prompt/相机变更重置。
- **动作语义**：`pi05_piper_dual` 使用 `use_delta_joint_actions=True`（模型内部 delta）；推理输出经 `AbsoluteActions` 还原为绝对角。RTC 的 `prev_actions` 需为 delta；若客户端只有绝对角，需按 `make_bool_mask(6, -1, 6, -1)` 转换（夹爪保持绝对）。
- **d/s 校验责任**：服务端做 sanity check + clamp（`d<0`→0，`s<d`→`s=d`，`s>H-d`→`s=H-d`）。若异常，记录 warning，必要时回退普通推理。
- **prev_actions 尺寸处理**：`action_dim` 不一致直接拒绝 RTC；长度不足右侧 padding（0 或 last action），长度过长截断保留最后 `H-s` 步。
- **rtc.reset 语义**：`rtc.reset=true` 清空 prev_actions/RTC 状态，默认不影响 prompt/观测缓存；prompt 变化可视为隐式 reset。更稳妥可引入 `episode_id` 做隔离。
- **能力元数据**：建议至少补 `action_horizon`、`action_dim`、`control_hz`、`use_delta_joint_actions`；在握手阶段发送。客户端无需每次上送 metadata。
- **适配范围**：先覆盖 `pi0/π0.5`（`pi0.py` 采样路径），再扩展 `pi0_fast`；给 BaseModel 增加 `sample_actions_rtc`，Policy 优先调用，未实现回退 `sample_actions`。
- **性能与稳定**：ΠGDM 引导用 `jax.vjp` 计算 VJP，采样循环用 `lax.scan + jit`；`β` 裁剪（典型 5），`τ` 设下限如 `1e-3`，必要时加 `grad_norm` clamp。若开销过大，可降低 `n` 或仅前 `k` 步引导；guidance 用 `float32` 更稳。

### 9.2 新增疑问 / 待确认
- **RTC envelope 版本化**：是否需要 `rtc.version` 字段，以便未来兼容不同 mask/引导策略？
- **action_horizon 来源冲突**：RTC payload 携带 `action_horizon` 时，是否必须与服务端/metadata 一致？不一致时如何处理？
- **并发请求策略**：同一连接是否允许未完成的推理叠加请求？若允许，如何保证 prev_actions 一致性？
- **时间戳字段**：是否需要在响应里回传 `server_time_ms`/`recv_time_ms` 便于客户端统计与排障？
- **相机 key 一致性**：`observation.images.<cam_name>` 的 cam_name 是否需在握手时白名单校验？

## 10. 服务端实施计划（本次范围）

### 10.1 协议与启动参数
- 在 `websocket_policy_server.py` 解析 `obs`/`rtc` envelope，剥离 `rtc` 后再构造 `Observation`。
- 在 `scripts/serve_policy.py` 增加 `--rtc-mode {off,auto,only}`，默认 `off`。
- 支持 `--rtc-metadata <path>`，在握手返回 `rtc_mode`、`action_horizon`、`action_dim`、`control_hz`、`use_delta_joint_actions`。
- RTC payload 与 metadata 冲突时：优先服务端 metadata；冲突记录 warning 并回退普通推理（可配置）。

### 10.2 RTC 推理入口
- 增加 `sample_actions_rtc(...)`，参数包含 `prev_actions`、`d`、`s`、`action_horizon`、`action_dim`。
- 处理 `prev_actions` 长度/维度：不足右侧 padding，过长截断至最近 `H-s` 步。
- `d/s` 约束：`d<0`→0，`s<d`→`s=d`，`s>H-d`→`s=H-d`，异常记录 warning。

### 10.3 采样引导集成（pi0 路径）
- 在 `pi0.py` 的采样循环中加入 ΠGDM 引导与 soft mask。
- 引导计算使用 `jax.vjp`，并对 `β`、`τ` 做裁剪以稳态收敛。
- 若开启 RTC 但采样失败或引导异常，回退普通推理并记录 error。

### 10.4 可观测性与回归验证
- 服务端日志输出：`rtc_mode`、`d/s`、`action_horizon`、是否 RTC/回退。
- `/healthz` 与握手保持原语义；RTC 元数据仅在握手返回。
- 添加最小验证脚本/用例：非 RTC 推理不变；RTC 请求返回动作且无断档。

## 11. 已完成的服务端改动（实现说明）

### 11.1 WebSocket 协议与回退
- 解析 `obs/rtc` envelope，剥离 `rtc` 后再构造 `Observation`；兼容旧协议（payload 直接是 `obs`）。
- `rtc_mode=off` 时忽略 `rtc`；`rtc_mode=auto` 尝试 RTC，失败/不支持则回退普通推理；`rtc_mode=only` 强制 RTC，缺失或出错返回错误。
- 响应里新增 `server_timing.rtc_used/rtc_warnings/rtc_error`，用于客户端判定是否发生回退与参数被 clamp 的原因。
- 代码位置：`src/openpi/serving/websocket_policy_server.py`。

### 11.2 RTC 参数校验与预处理
- 校验 `d/s`、`action_horizon/action_dim`，并做 clamp；异常时记录 warning 或回退（视 `rtc_mode`）。
- `prev_actions` 支持空数组；不足长度右侧 padding、过长截断保留最近 `H-s` 步。
- `rtc.reset=true` 时忽略 `prev_actions`，以确保新 episode 或 prompt/camera 变化时不串状态。
- 代码位置：`src/openpi/serving/websocket_policy_server.py`。

### 11.3 Policy 与模型侧改动
- `Policy` 新增 `infer_rtc`，优先调用 `sample_actions_rtc`；未实现时抛出 `NotImplementedError` 由服务端回退。
- `pi0.py` 增加 `sample_actions_rtc`：ΠGDM + soft mask，引导项用 `jax.vjp` 计算 VJP，`β/τ` 做裁剪。
- `serve_policy.py` 握手 metadata 自动补齐 `action_horizon/action_dim`（从模型配置读取），并合并 `--rtc-metadata`。
- 代码位置：`src/openpi/policies/policy.py`、`src/openpi/models/pi0.py`、`scripts/serve_policy.py`。

### 11.4 旧推理与 RTC 共存保证
- 旧客户端仍可发送裸 `obs`；服务端保持原 `policy.infer` 路径不变。
- 新客户端可按 `rtc_mode` 决定是否走 RTC；`auto` 支持灰度与回退，不破坏旧链路。

## 12. 客户端需要做什么（更新）

### 12.1 连接与握手
- 连接后读取握手 metadata，确认 `rtc_mode` 与 `action_horizon/action_dim/control_hz`。
- 若 `rtc_mode=off` 或握手缺失，则退回旧协议（裸 `obs`），保持兼容。
- 可在本地提供 metadata override，仅用于兜底，不上送服务端。

### 12.2 RTC 请求内容
- 采用 envelope 发送：
  - `obs`：原观测数据。
  - `rtc.prev_actions`：长度为 `H-s` 的**绝对关节角**动作序列。
  - `rtc.d`：基于 RTT 估计的延迟步数（`ceil(RTT / Δt)`）。
  - `rtc.s`：执行步数，建议 `s = max(d_est, s_min)`。
  - `rtc.action_horizon/action_dim`：用于 sanity check，必须与握手 metadata 一致。
  - `rtc.reset=true`：在新 episode、prompt 变更或相机变更时显式重置。
- 动作语义必须与服务端一致（`use_delta_joint_actions=True` 时发送绝对角）。

### 12.3 prev_actions 与执行对齐
- `prev_actions` 必须来自**实际执行过的动作**（不要用“预测但未执行”的动作）。
- 初次请求或重置时可发送空数组，服务端负责 padding。
- 若客户端允许并发请求，必须确保 prev_actions 与执行顺序一致；不确定时改为串行。

### 12.4 客户端统计与回退
- 维护滑动窗口统计 RTT（如 `b=10`），取 `max` 或 `p95+1` 作为 `d_est`。
- 若收到服务端 RTC 错误或回退标记（`server_timing.rtc_used=false` 或 `rtc_error`），记录日志并退回普通推理。
- 注意异常帧可能是纯文本错误（非 msgpack），客户端需能识别并处理。
- 可选：记录 `d_obs`、`rtc_warnings`、动作断档率，便于调参。
