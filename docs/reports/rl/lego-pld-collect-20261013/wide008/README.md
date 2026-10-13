# Actor-1 / Collect-2：扩大探索，保留学习中心

用户授权先试每关节±0.08rad，首集核查后采到10–15集，再训练Actor-2。当前权重仍是`PARTS-TD3BC-Balanced seed0/3000`（500次actor更新），只新增采集策略；不把改噪声称为训练出新actor。旧13集及第14/15集保留。

## 数值选择

固定原13集467次合法抓取、2,950个可控上下文，每处64个共同随机样本（seed138）。在Thor CPU推理冻结actor作为中心，计算8步包络内最大目标FK差，不训练或连接机械臂。188,800个样本不是188,800次真实尝试，相邻姿态不是独立实验重复。

| 配置 | 中位目标偏移 | P95 | P99 | 本次最大样本 |
|---|---:|---:|---:|---:|
| 旧collect：±0.02rad/std0.2/增量0.008 | 3.09mm | 6.33mm | 7.94mm | 13.25mm |
| 新collect：±0.08rad/std0.5/增量0.03 | 21.56mm | 42.42mm | 51.92mm | 82.43mm |

新配置90.60%样本超过1cm、7.06%超过4cm、1.43%超过5cm；左/右中位20.73/22.11mm，P95为40.96/43.15mm。六关节可行域无缩小；峰值关节RMS为0.03340rad。上下方向均有覆盖，最大位移时刻竖直差P5/P95为−26.22/+28.27mm。闭合前/后均覆盖厘米级。完整SHA和分组见[几何复核](geometry-all13.json)。此前139上下文提案保持原件。

±0.08是相对Pi参考关节目标的总偏移边界（约4.58°/关节），不是相对actor中心再加±0.08或每帧累加0.08。4.24cm是P95，不是每次都动4–5cm；5cm不是末端硬上限。FK不证明真实跟踪、接触安全或抓取改进。8步包络约0.267秒，按`[0,1/3,2/3,1,1,2/3,1/3,0]`淡入淡出，不能宣称维持4cm直到完成抓取。

[PARTS附录D](https://arxiv.org/html/2609.21788v2)的LEGO示例中±0.50只用于归一化夹爪开度，不能与六关节0.08rad比较。本项目已获用户确认修正整段抓取的六关节与夹爪，是扩展而非该配置的逐项复现。

## 实现及数据坐标

新快照`23d4da1ad778586cfcafdb02868eec7e51810d50caed259c4e4a4c5184383efd`，合同`9b2a4ab59b7810bdbd3148b07ceb43880034910d937c046dbf39085f773c87fe`；父快照`75674aad…`，actor权重SHA始终`8cd1a5a0…`。[manifest](collector-manifest.json)绑定原训练配方、标准化、编码器和FK。

- 根`contract`保存新采集坐标，`actor_contract`保存原0.02坐标。两者只允许radius、残差步进和说明不同。
- collect先以原坐标得到actor建议的物理残差，再按当前正负可行边界转换成新tanh潜变量中心，加入六关节std0.5、夹爪std0.2高斯噪声。夹爪半径±0.05、增量0.02保持；关节增量guard0.03避免旧0.008把大探索压小。SDK绝对限位、总目标步进、RTC前缀保持。
- 零噪声输出保持父模型物理建议，所有原上下文解码最大差3.73e−9rad；非线性tanh后分布的数学期望不保证与中心严格相等。
- eval/probe/shadow使用父合同，eval不把学到的偏移乘4。握手绑定整个新bundle和根合同，trace另写`execution_contract_sha256`。
- trace的`x/action/scale/negative_scale`为实际collect坐标；新增`actor_context/actor_mean_u/actor_scale/actor_negative_scale/actor_physical_center/collector_latent_center/collector_latent_std`可重建旧模型输入与新采样中心。图像/状态/计划与x共享，不重复传输整个输入。
- Replay检查新坐标解码、原生实际下发、中心和旧输入context，奖励仍归同一次抓取的合法结算终点。合成录制→切段→奖励→Replay通过。新bundle的probe用父坐标，不混进扩大坐标Replay。
- 下轮混入旧13集须按物理残差转换归一化坐标、重建边界/输入并处理模型warm-start，不能直接拼不同半径的action数组。此迁移留给下轮训练准备，当前未启动新训练。

## 验收与采集

CPU合同/抓取/录制回归29项通过。Thor CPU真实RGB＋记录Pi输出4例、collect12次验证完整ResNet18和actor路径，eval输入/动作逐值一致、噪声中心与父动作一致、前缀保持；[CPU检查](runtime-cpu-check.json)。候选待完整Pi＋RL GPU延迟、客户端包验证与生产切换，线上仍是旧Actor-1 collect配置；状态以部署回执为准。

确认现场停止后再切换同一`pld-rl`地址，旧容器保留回退。客户端集间重新绑定身份，前缀仍10。首集检查快照、有效修正tick、实际指令差、FK分布、机械臂响应、延迟/取消、两臂触发和奖励覆盖；如果队列覆盖或机构未跟上，先修执行时间/节奏，不能只加radius。通过后补齐10–15集，Qwen＋DINO判奖、无建议交人工，冻结奖励版本再训练Actor-2，与父Actor-1/纯Pi对照。
