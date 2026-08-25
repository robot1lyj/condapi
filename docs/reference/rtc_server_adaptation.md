# RTC 服务端参考

这是 RTC 的实现边界参考，不是 OpenArm rollout 操作手册；实际执行先读 `docs/05_inference_and_rollout.md`。

## 模式

`scripts/serve_policy.py` 支持：

```text
--rtc-mode off   忽略 RTC payload，保持旧推理路径（默认）
--rtc-mode auto  有合法 RTC payload 才使用，否则回退旧路径
--rtc-mode only  必须有合法 RTC payload，否则拒绝请求
```

服务可用 `--rtc-metadata <json>` 在握手 metadata 中附加合同信息。metadata 只能描述当前 checkpoint 的真实输出，不能用 Piper 模板冒充 OpenArm。

## OpenArm 安全要求

- 先用 `rtc-mode off` 完成 baseline 与真实 WebSocket smoke，再单独验证 `auto/only`。
- OpenArm 输出为 50×16、degree/HQ 夹爪 `0/-66`；客户端才做 ROS 弧度/归一化转换。
- `prev_actions` 的 delta/absolute 语义必须与当前 config 的 action transform 一致；不能重复做 delta 累加。
- RTC 失败时，`auto` 必须回退旧路径；`only` 的拒绝要被客户端识别并安全停机。
- 每次 RTC 试验记录 config、checkpoint、metadata、client payload、mode、动作摘要和回退次数。

## 兼容性验收

任何 RTC 代码改动都要同时验证：旧 `off` 请求、合法 `auto` 请求、非法/缺失 payload 的 `auto` 回退、`only` 拒绝，以及动作形状/有限值/单位 metadata。通过后再进入真机 rollout。
