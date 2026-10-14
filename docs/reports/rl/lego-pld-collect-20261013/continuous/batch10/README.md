# 连续探索10集 / Actor-2训练与上线

用户2026-10-14确认10集录完并授权暂停Thor推理、判奖及训练。数据来自同一session的000001–000010，快照7013be46…、关节±0.08rad连续探索、前缀10，无人工介入。第一集为开发验证，其余9集训练，不把已调试的第一集称独立未见测试。主Pi与编码器冻结，旧13集及Actor-1保持原件，首次Actor-2训练不直接拼入旧动作合同。

10集35,724行/1,190.71秒，106个原件复制前后及本地/Thor SHA核验通过，72段视频完整解码；3368个接受回复，0实际提交差异、0来源缺失。操作员整轮标签为9/7/8/7/10/9/9/7/7/6，共79/100，1/10完整完成。这是collect行为数据，包含探索噪声，不能当确定性actor能力测试。

## Actor现场指标

固定N=10，先报告C/N与完整完成率。完整完成的轮次比较抓取尝试数A与耗时；可以附加效率分`min(1,10/A)`，未完成记0，避免提前停录的轮次因次数少占优。`C/A`只描述整任务完成效率，不叫局部抓取成功率。

本场景A的可复算口径：沿用relative_v3实际夹爪开合迟滞切段，闭爪或参考点FK位于已标定抓取空间时计入；空抓仍计入。431个候选中430个物理收拢、1个仅预测窗口；14个收拢在抓取空间外，逐一查看录像后确认E04-R022-v3是抓取空间外捡回黄色积木，整任务计数补回1次；另13个为容器/运输动作。最终计数共417，逐集36/44/23/51/43/49/31/50/36/54。它是固定规则计数，不冒称逐次人工真值或通用阶段识别。保留每个计入/排除决定及14份候选SHA绑定的视觉计数复核；它不修改RL奖励。[逐集表](episodes.csv)、[计数复核](count-reviews-v1.json)。更改规则必须对各策略统一重算。当前唯一10/10的第5集用了43次收拢，效率10/43；未完成轮次保留C、A、耗时，不按A独立排序。

## 奖励与补充模型的限制

主Qwen＋DINO共242个合格终态提案，held97/empty145；增加10个视觉明确肯定抓取意图且完整运动证据确认未达到抬升目标的not_lifted=0。252个发布窗口，1658个原始区间，62个无可用建议保留待复核，不强制写0。Replay-v2最终接受245次尝试/1635个执行区间；7个奖励窗口因为没有实际非零修正不入库。合并强制等待尾段并保持折扣回报后为882个学习决策：228次尝试/825决策训练，17次/57决策开发验证；左右82/163。训练集held88/failure140，验证held5/failure12。62个无建议不阻塞当前训练，也不被强填0。[冻结输入摘要](training-input-summary.json)。

另做431例Qwen抓取意图探针，False274/True125/uncertain32。主判奖已提议成功的97例中，意图探针仍给出35个False，理由常为“指间无物体”；它混淆空手与没有尝试，不能用来排除空抓或直接减少A。False/uncertain不排除数据；held_then_lost也未用于自动补发掉落奖励。原始探针及失败结果全部保留，不作为已验证奖励模块。补充失败只使用正向意图证据与现有运动合同共同支持的10例，不声称新增了10个人工真值。

## 本轮训练配方

继续当前PLD外挂框架的PARTS式TD3＋成功BC学习器，非论文完整复现。三组各seed0/1：`PARTS-CT-TD3BC`（Q系数0.1/失败零锚定0.1）、`PARTS-CT-NoAnchor`（Q系数0.1/零锚定0）、`BC-CT-Success`（Q系数0/零锚定0.1）。第三组用于检查RL是否超越单纯拟合成功修正；不把不同loss直接排名。

统一3×256网络、FP32、batch256、lr3e-4、3000次critic更新、前1000次MC校准、前2000次critic预热、最后1000步每两步更新actor，共500次RL/BC更新。另在前200步用记录的Actor-1行为中心做新坐标对齐初始化，只使用训练集，不移植旧critic，也不直接把旧输出放大4倍。原始执行action继续是critic监督，行为中心仅作初始化目标。五组输入RMS仅由训练集计算；归一化、冻结ResNet18与Pi身份不变。

新actor以signed_trajectory_v1坐标训练、导出和推理，eval与collect都使用新解码器及同一±0.08目标上限；collect另加相关噪声。不能沿用Actor-1的±0.02输出映射。所有检查点、优化器、RNG、指标、导出包和失败尝试保留Thor；工作站仅做CPU数据处理/协议检查，不训练。

## 训练结果与选择

训练代码26c1b76，Thor独立容器镜像`sha256:325ef815f1ae6b153c04425ff96d55595c2d17a86ee29b56cbe5752201780afd`。六组18,000 critic/3,000 RL或BC actor/1,200坐标对齐更新完成，36检查点均通过有限值、导出一致性和验证复算审核；更新、评估、保存循环合计202.09秒，不包含视觉判奖/数据准备/模型启动。[六组原始结果](round03-results.json)、[训练完成回执](training-complete.json)。这些完成回执在上线前冻结，其中`production_actor_replaced=false`描述当时状态，当前上线另见deployment.json。

| 组别（seed0/1） | 入口AUC | 入口记录回报RMSE | 全验证目标RMS | 训练各维1–99%范围外比例 |
|---|---|---|---|---|
| TD3BC | 0.70 / 0.65 | 0.572 / 0.578 | 0.219 / 0.215 | 0 / 0 |
| NoAnchor | 0.65 / 0.60 | 0.687 / 0.711 | 0.474 / 0.439 | 0.130 / 0.080 |
| BC对照 | 0.80 / 0.783 | 0.456 / 0.467 | 0.189 / 0.198 | 0.0075 / 0.0025 |

选择有失败约束的TD3BC seed0/3000作工程测试候选，BC seed0/3000保留做同条件对照；NoAnchor更偏离行为范围，不优先上线。17次开发验证较少，全段AUC也接近随机，critic价值仍偏高；不能声称RL已优于BC、收敛或提高真机成功率，Q自评gain不作为选择效果证据。首次测试是检查新版确实能平稳执行并取得真实结果，然后做Pi/BC/TD3BC对照。

## 几何与接口验收

冻结actor仅前向，无训练：882个记录上下文，862处可编辑。TD3BC目标轨迹相对Pi的峰值FK差中位10.38mm/P95 25.68mm/最大42.10mm，峰值关节偏移中位0.02493rad/P95 0.05895rad/最大0.07144rad；BC对应FK差中位3.43mm/P95 25.45mm/最大42.31mm。两者前缀、退出尾段、范围通过。这是记录上下文的目标几何，不是新actor真机位移或反事实成功证明。[TD3BC](td3bc-geometry.json)、[BC](bc-geometry.json)。

两候选均独立GPU16例、完整Pi＋专家48例/普通RTC9例预验收、原客户端校验器16包通过。新原生actor的probe/shadow旁路桥接、逐值返回Pi；eval/collect保持不可编辑前缀与非活动臂。推理源码c7c4e54，训练后仅修复诊断模式和验收逻辑，权重未改。[TD3BC GPU](td3bc-runtime-gpu.json)、[BC GPU](bc-runtime-gpu.json)、[生产Pi＋专家](production-bundle-smoke.json)、[生产RTC](production-rtc-smoke.json)、[原客户端验证](production-client-validation.json)。

## 当前上线与现场操作

2026-10-14 14:13 CST，用户本轮已授权现场暂停、判奖训练，随后要求上线测试，正式容器`pld-rl`切到TD3BC Actor-2，healthy/MAXN、会话单元active。生产推理P50/P95=218.02/219.33ms。验收为Thor本机真实模型和原客户端协议检查，尚未含新actor真实3588执行；新的第11集eval已核查，真实跨IPC执行通过，发现临闭合时短窗口提前撤掉修正及夹爪幅度不足的改进点；不混入本轮训练。[逐项证据与建议](eval-e11.md)。

- 地址仍为`ws://192.168.250.1:8000`；重连后重新选eval、前缀10，先1–2集确定性现场测试。
- actor：`parts-ct-td3bc-r03-s0/bundle-003000`，500次RL/BC更新＋200坐标对齐；eval/collect均用新0.08轨迹坐标，collect另加噪声。
- 快照`f668170b8c59bd799318f20bdf338a5349dfdc31295058dff21ae3eeac88a24c`；actor SHA `bffec6f3f29e10ab529acaae3794c134076b34e6a092bf291cbe6c7c3cb09b76`；合同`959db331948a9acd7c59f45bae64cc6298bcb04c879e30ea414ef5de465bba9e`。
- 镜像`pld-rl:thor-actor2-continuous10-20261014-r03`，ID`sha256:0abd010efa86b9339462d4ea03a98a145829581962c7217bb7f94a28df4de3ef`，MAXN单元`thor-pld-actor2-continuous10-20261014-r03-maxn.service`。
- Pi20w/7步FP32权重与TF32 TRT/quantile RTC不变。部署回执[deployment.json](deployment.json)，选定导出合同[TD3BC](bundle-td3bc-manifest.json)、[BC](bundle-bc-manifest.json)。`eval_preserves_parent_target_scale`沿用了旧说明字段，实际尺度以actor/eval合同与radius为准，不能解释为新actor仍是0.02。

旧生产Actor-1停止保留为`pld-rl-retained-pld-actor2-continuous10-20261014-r03`。回退需先停现场执行、停止新容器并退出其MAXN会话，换回保留容器及其MAXN会话，重新核对7013…身份；不能在一集内混换策略。BC候选完成预验收，未切生产。

## 原件与可复算归档

- 本地数据`/home/wuyan-lyj/condapi-data/rl/lego-pld-continuous10-20261014-v1`，Thor训练数据`/home/wuyan-lyj/thor/rl-training/datasets/continuous10-20261014-v1`，含原件/哈希、curator-v1、vision-combined-v2、reward-release-v1、replay-v2、learning-replay-r03-v2、training-input-v1与行为bundle。
- 全量Qwen＋DINO/补充意图探针输入输出在Thor`/home/wuyan-lyj/thor/parts-vision/datasets/lego-pld-continuous10-20261014-v1`及`…-intent-all-v2`。失败探针保留，不当作有效计数器。
- 六组所有保存检查点、优化器、RNG、metrics、预测和导出在Thor`/home/wuyan-lyj/thor/rl-training/runs/<run>`；审计/编排在`runs/continuous10-r03-orchestration`。训练代码固定在`rl-training/releases/26c1b76`。
- 上线源包、两候选bundle、GPU/协议/性能回执及原返回包在Thor`/home/wuyan-lyj/thor/pi/services/pld-actor2-continuous10-20261014-r03`；BC完整预验收在`…/pld-actor2-bc-continuous10-20261014-r03`。本页只入Git小型摘要/身份，原始视频和权重不入Git。
- 第一次Replay-v1因独立浮点重建约1e-16差异错拒大量合法区间，保留失败原件；v2将精确wire检查绑定录制物理动作＋原前缀，重建仍保留1e-6容差，未放松实际下发检查。Actor-1训练后恢复时观察器一度先看见Exited、随后systemd已正常启动；原恢复日志及RTC9例通过回执均保留，最终生产已由Actor-2替换。

训练看板沿用8766六组曲线，复核沿用8768固定模板；无建议62例可之后补审，不影响已冻结训练。下轮新增奖励/经验需新版本，不覆盖本轮原件。
