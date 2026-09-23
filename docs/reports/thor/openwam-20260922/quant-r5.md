# OpenWAM-α 历史 FP8 探索 r5

2026-09-22 Thor MAXN，官方 RoboTwin-Full BF16 检查点原件不变。使用 TorchAO `Float8DynamicActivationFloat8WeightConfig` 对现有训练权重做选择性动态 FP8，保留动作分支和归一化。三组合成三相机/20D状态/prompt输入；每组一次预热，随后9次完整 `engine.generate` 计时。探针看到实际 `aten::_scaled_mm`，全部输出有限 `(32,20)`。数据与脚本：[原始JSON](quant-r5.json)、[`quant_benchmark.py`](../../../../scripts/thor/openwam/quant_benchmark.py)。

| 配置 | P50/P95 | 与本轮 BF16 参考的全维 MAE / 最大差 |
|---|---:|---:|
| BF16 | 1264/1276ms | 0 / 0 |
| 视频 FFN 60个Linear FP8 | 1307/1307ms | 0.00191 / 0.02344 |
| 视频块300个Linear FP8 | 1726/1748ms | 0.00247 / 0.03320 |
| 视频块FP8＋编译 | 824/828ms | 0.00345 / 0.05273 |

不编译时量化反而变慢；编译组冷调用约275秒。这里的最大差混合EEF位置、rot6d和夹爪单位，不是关节弧度；三组合成输入不能验收任务精度。报告没有序列化量化权重或TensorRT引擎，不能声称模型磁盘体积已减半。按用户后续指示，当前优先 BF16 官方路线，r5 仅保留历史证据，不作为部署候选。
