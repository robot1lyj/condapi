# PLD-Grasp 学习候选上线：PARTS-TD3BC-Balanced

2026-10-13，用户明确要求恢复原`pld-rl`供现场测试。已完成候选与生产各48次真实录像输入的完整Pi＋专家验收，另各9次普通RTC；本地只读复用YAM客户端校验器，各16条回复通过。实际生产`pld-rl` healthy、MAXN，地址仍`ws://192.168.250.1:8000`。这是Thor本机验收，未冒充新版本跨IPC/机械臂闭环验证，待用户重连现场测试。

- 专家：`parts-td3bc-balanced-collect13-r02-s0/bundle-003000`，3000 critic / 500 actor更新。Pi仍为50h、200000、7步FP32权重/TF32 TensorRT、quantile trained RTC。
- 快照：`75674aad5775e4864cce24664abd270e43a2a82b7e64f95811ce952979b2d286`；合同`87b1affa6d81e1d0cf6299bf6e827bd260361104338708e67579c874ee65d7ec`不变。
- 镜像：`pld-rl:thor-balanced-20261013-r02`，`sha256:f6dfaba965ec61b4022cc4b26cdda420ac3de4b79ccec42687dcf84f35295bd0`；运行时代码`f6729d8`，训练代码`70184d5`。
- 生产服务P50/P95：213.64/216.96ms，原先约213/214ms；不是IPC往返时延。12个eval用例实际输出非零学习修正；probe/shadow不修改物理计划，RTC前缀逐值保持。
- 关节±0.02rad、夹爪±0.05，原先动态可行域/速率与门控保持；冻结编码器和Pi无新权重、精度或引擎转换。
- 旧服务保留为`pld-rl-retained-pld-balanced-20261013-r02`，原bootstrap包未覆盖；训练容器`pld-rl-train`停止，避免与在线推理竞争。

首轮先重连并重新应用`eval`（已学习均值，无额外随机探索）＋前缀10，录2集；对照用`probe`零修正，同分布交替采纯Pi与候选各2集。原配置若固定旧快照，需重新应用新握手身份，不在任务中途静默换专家。eval不进入现有训练Replay；验证执行/方向后才用collect采下一批训练经验。右臂入口判断仍弱，尚未证明降低空抓或整轮成功率提升。

完整产物：Thor `/home/wuyan-lyj/thor/pi/services/pld-balanced-20261013-r02/`；工作站 `/home/wuyan-lyj/condapi-data/rl/lego-pld-collect13-20261013-v1/deployment-r02/`。原始完整返回包、构建/切换日志和失败尝试均保留；没有向客户端agent发任务，没有改3588控制/相机/系统。

[生产回执](deployment.json)、[学习专家验收](production-bundle-smoke.json)、[普通RTC](production-rtc-smoke.json)、[客户端协议校验](client-production.json)、[2950个真实记录上下文解码](recorded-context-check.json)、[训练对照与选择理由](../round02.md)。
