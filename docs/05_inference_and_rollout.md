# 05 · 推理与 Rollout

## 1. 服务启动

OpenArm checkpoint 服务使用 `policy:checkpoint`，并把 `--port` 放在它前面。K-Policy 必须覆盖客户端 prompt：

```bash
cd /share/home/linyongjia/conda-pi/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
/share/home/linyongjia/miniconda3/envs/pi-conda/bin/python scripts/serve_policy.py \
  --port 6666 \
  --force-prompt 'Fold the T-shirt properly, Advantage: positive' \
  --rtc-mode off \
  policy:checkpoint \
  --policy.config=pi05_openarm_kai0_awbc_v1 \
  --policy.dir=<CHECKPOINT_DIR>
```

`default_prompt` 只在客户端没有 prompt 时生效，不能替代 K-Policy 的 `force_prompt`。启动前核对 checkpoint、config、数据 repo 和 norm stats 是同一套产物。

## 2. 真实 WebSocket smoke

```bash
conda run -n pi-conda python scripts/smoke_test_openarm_policy_server.py \
  --host gpu25 --port 6666 \
  --checkpoint <CHECKPOINT_DIR> \
  --output /tmp/openarm_policy_smoke.json
```

通过条件是：真实请求成功；动作有限且形状 `(50,16)`；metadata 确认 50-step、16D、degree、HQ 夹爪 `0/-66`；报告绑定实际 checkpoint 和 prompt。只有端口监听、进程存在或能加载模型都不算部署成功。

## 3. Rollout 分层

```text
静态检查 -> 仿真/离线回放 -> 低风险真机单步 -> 小批量固定协议 -> HIL raw -> clean/export -> 训练/再评估
```

每一级都保存 checkpoint、config、prompt、数据版本、节点、时间和日志；不跨级跳到连续真机。真机开始前确认急停、工作空间、夹爪限位、动作频率和人工接管通道。

## 4. HIL 采集

- 固定 collector checkpoint 和强制 prompt；不要在同一批数据中途切换 20k/79999 或其他模型。
- Raw 保留策略失败前缀、policy action、human action、hold、intervention、视频/时间戳以及 `episode_success`/`recovery_success`。
- clean 阶段丢弃 hold 等等待帧，保留真实 human VR correction；逐集检查 16D、单位、时间同步、视频尾帧和成功结尾。
- 当前研究计划中的 HIL-T30 是错误对角线、重复甩平、已展开不折叠三类恢复各 10 条；是否已完成必须查远端数据，不看计划文字推断。

## 5. Rollout 指标

至少按固定任务集统计：正确对角线选择率、无进展重复甩平率、已展开后进入折叠率、完整折叠成功率、每集接管次数、恢复成功率和动作/延迟异常。训练 loss 或单条成功演示不能替代真机结论。

## 6. RTC

服务支持 `--rtc-mode off|auto|only`，默认 `off`。OpenArm 先用旧路径完成 baseline；启用 RTC 时单独记录 metadata、`prev_actions` 的 delta/absolute 语义和客户端 payload，并保留 `off` 回退。任何 RTC 修改必须同时通过旧路径和 RTC 路径 smoke；不得把 Piper 专用 metadata 当 OpenArm 合同。

## 7. 停止条件

出现非有限动作、形状/单位不符、prompt 未强制、视频/时间戳不一致、急停或接管失效、checkpoint 不完整时立即停止 rollout，保留 raw 和日志，修复并重新 smoke；不要用“看起来能动”继续采集。
