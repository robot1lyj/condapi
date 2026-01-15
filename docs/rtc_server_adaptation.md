# OpenPI 服务器端 RTC 适配指南（pi05_piper_dual）

> 目标：在 `openpi` 服务端实现 Real-Time Chunking（RTC）推理，使远程推理在高延迟下仍连续、平滑。当前服务端启动命令：
> 
> ```bash
> python scripts/serve_policy.py \
>   --port 6666 \
>   policy:checkpoint \
>   --policy.config=pi05_piper_dual \
>   --policy.dir /output/openpi/pi05_piper_dual/piper_ft_pi05/4999
> ```

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
- **模板参考**：`/home/lyj/orin_VR/inference/remote_infer/rtc_metadata_template.json`
- **相机信息来源**：`/home/lyj/orin_VR/configs/piper_inference_dual.json` 中 `cameras` 字段。

示例模板字段（节选）：
```
{
  "model_name": "pi05_piper_dual",
  "action_horizon": 50,
  "action_dim": 14,
  "control_hz": 30,
  "use_delta_joint_actions": true,
  "action_units": "absolute_radians",
  "input_keys": ["observation.state", "prompt", "observation.images.<cam_name>"]
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
- **动作格式**：`use_delta_joint_actions=True` 时，模型输出已经是**绝对关节角**。`prev_actions` 也必须是绝对角，否则会出现“二次累加”。
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
- **动作语义**：`pi05_piper_dual` 使用 `use_delta_joint_actions=True`，推理输出已还原为**绝对关节角**；`prev_actions` 必须同语义。夹爪保持绝对（`make_bool_mask(6, -1, 6, -1)`）。
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
