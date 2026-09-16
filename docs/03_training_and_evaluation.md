# 03 · 训练与评估

> 当前路线：用户于2026-09-08明确切换为 **Pi0.5全量微调**，正式训练已启动。
> 旧LoRA低显存默认建议不再适用，不自动回退；保留的LoRA代码和历史记录不代表当前实施方案。
> 2026-09-16 当前状态：用户因效果不佳暂停训练，完整可用基线为100000；下方80k/324194目标和旧保存间隔是历史计划。下一轮设计见本页“训练重设计”；本轮只读核查和制定计划，没有恢复训练、替换在线模型或切换默认训练方式。

## 2026-09-16 · 训练重设计：先复现乐高分拣，再比较模型

### 目标、现场与已知进度

用户报告100000模型能接近目标，但抓取或后续步骤失败；确认现场与示范基本一致，使用ABC-130K同款官方YAM kit、三台D405、640×480，近期优先复现原数据任务。已发布数据唯一任务为 `sort the legos into containers by color`，本轮验收据此定义为抓起并按颜色放入容器。现场安装几何、物体像素尺度和完整任务成功率尚未独立量化。

**后续更正与当前决策：** 用户确认现场普通2×2积木底面约16 mm，并判断换用了不同于ABC的积木。六集训练帧抽查支持物体形态/相对尺度存在差异，但没有源物体实测尺寸或全量比例。现场路线为冻结并验收执行时序→少量现场专家数据→以100000权重另开现场适配run。用户随后提出利用采集窗口训练Evo，建议并行准备D1上的Evo小预算试验，见下方“采集期间的Evo-1四卡试验”；与现场工作并行，不与Pi训练同时占满四卡。补采和新训练均未执行。

服务器只读快照见 [证据](reports/training/redesign-20260916/README.md)。训练于13:58:20 +08按用户要求暂停，最后日志100531，最新完整checkpoint仍100000；14时段本用户Slurm队列为空。暂停前从干净100000恢复的新run目标已改为120000，batch32/FSDP4、workers2、peak LR 1.25e-5、Adam eps 1e-6；改动和约531步有限日志不证明NaN根因已解决。旧链中110000及后续抽查点的词嵌入NaN见 [checkpoint审计](reference/thor/12_checkpoint_handoff.md#2026-09-15--158000首次实际接入结果)，不可作为“多训后的效果”对照。

100000全参数有限、JAX↔FP32转换及9输入TensorRT对照已完成，见 [100000报告](reports/thor/100000.html)。这些对照没有测专家动作误差、全任务成功率或实时重规划一致性；加速误差对抓取的影响仍需专门对照。

### 数据量与时间的口径

train为4458集、10,374,181帧、30fps、1个task；val为69集、165,142帧。train约96.06小时，含val约97.59小时。“80小时”不作为下一轮预算分母；按固定版本实际帧数计算。

`100000 × 全局batch32 / 10374181 ≈ 0.3085`，即约30.85%的一遍样本量，不是不到10%。这是等效训练量估计；历史链含配置变更/恢复，确切值须累计各段实际全局batch、接受的更新和采样位置，不能当独立样本覆盖率。FSDP4不再把全局32乘4；H50标签重叠，不能再乘50解释为独立覆盖。随机帧采样也不等于只训练了前31%的episode。

当前loader按epoch洗牌并按step恢复位置，`drop_last=True`，本帧数/batch下每epoch为324193个完整batch，丢最后5帧；历史324194是ceil预算近似，不是每帧均已训练的保证或收敛门槛。

暂停前近期约2.294秒/步、约14起点/秒。按2.3秒/步估算：5k约3.2小时，10k约6.4小时，20k约12.8小时，100k约63.9小时；不含编译、保存、故障和评估。缩小数据集通常不改变同模型/同batch的单步计算量，主要增加相关示范重复频率、减少取得结论的总步数。2小时×30fps=216000帧，在batch32下一遍6750步，约4.3小时；不要求恰好一遍才评估。

### 优先验证的失败机制

**最新为14:51会话，夹爪已能闭合。** 1370帧/45.755秒、1364有效policy行、130回复；左右反馈最低0.0818/0.0387，反馈<0.3分别379/264行。视频抽查出现目标靠指尖一侧、闭合后再次张开且物体仍留桌面的片段；没有经标定的接触/毫米误差或完整任务成功率。仍因录制队列满结束，原始数值/三路视频哈希、接触前后抽帧及六集ABC对照归 [最新证据](reports/training/redesign-20260916/README.md#乐高分拣3最新异步测试)。不能继续以“夹爪始终关不上”为当前根因；底层时序仍须验收，见下节。

**14:14会话保留作历史对照。** 用户补充已优化为异步推理、关节动作改善但夹爪不能闭合，指定查看“乐高分拣3”。当时经只读状态接口定位到 `session_20260916_141405_f47544/episode_000001`，只取记录文件，未操作机械臂、相机或控制实现。已核对源端/本地HDF5哈希一致，分析见 [历史证据](reports/training/redesign-20260916/README.md#1414会话历史异步夹爪问题)。下方90帧属于更旧版本。

- 1325帧、44.26秒、126个有效回复；manifest为streaming=true、RTC=false、ensemble/3块。热态观测到回复p95为164.6ms，选中动作索引4–15。记录结束为 `episode queue full` / aborted，不是完整任务成功率样本；当前状态接口100Hz轨迹设置不能独立证明整段每次电机写入情况。
- 有效帧左右夹爪策略最低分别0.473/0.507，下发最低0.813/0.671，反馈最低0.817/0.677（各列最低值不一定同帧，0闭/1开）。右夹爪17/126个完整H50块存在<0.3的远期预测，实际选中动作没有<0.3；第100请求首次<0.3在索引35，记录仅执行索引4–14后被新块接替。
- 选中夹爪目标与相应原始块索引值基本相同（左最大差0，右约0.000142），不支持本记录中夹爪先被三块ensemble平均的猜测。模型选择与后续下发仍有明显差距：38.79秒同帧右侧策略0.507→下发0.737→反馈0.756；左侧5.03秒策略0.473→下发0.828→反馈0.846。
- **结论限于记录：** 闭合目标未持续进入近期执行区间，且短暂近期闭合被提交环节进一步削弱；应同时查重规划一致性和执行侧约束/轨迹延迟。已有部分位置的反馈跟随不证明全行程映射正确，不能断言方向/比例一定正常，也不能直接将模型远期闭合提前执行、反转夹爪或强制二值化。本轮没有改控制代码，具体哪条执行限制产生差值仍待YAM侧按此回执定位。

14:14提出的P0是核对原始H50、observation时刻、选中索引、策略值、提交目标和反馈；14:51已通过持续闭合这一小项，但不等于完整映射/抓取通过。当前P0转向下节的手臂与夹爪相位、完整录制和任务验收。模型侧固定输入/噪声消融与低延迟参考对照仍用于区分策略重规划和执行转换，不能用延长整块盲执行掩盖问题。

1. **预测与执行分开检查。** YAM本地 [验收记录](/home/wuyan-lyj/YAM/yam-abc-reproduce/docs/acceptance.md) 的203帧短测显示夹爪0闭/1开映射有端到端证据，近期模型目标仍偏开；190/188帧测量显示相邻块不一致与跟随限制同时存在。放宽包络后方向反转增加，不支持直接以放宽限制解决。本轮仅引用记录，未读改3588控制/采集实现。
2. **独立复核90帧原始记录。** [重算结果](reports/training/redesign-20260916/trace90-summary.json)：13回复、约2.96秒，热态观测到回复p95约167ms；有效帧右夹爪近期策略最小0.769、下发最小0.854、反馈最小0.859，完整H50预测最小0.181。70/90帧至少一关节命中约束。应检查闭合是否随重规划后移及策略/反馈偏离，但短片段无专家目标、RGB或接触标签，不能证明完整抓取失败原因。
3. **分离采样噪声与输入变化。** Thor服务每请求重新生成噪声，既有离线对照固定噪声。下一步在Thor离线固定真实observation，分别固定噪声重复、改变噪声重复；再对连续观测检查同目标时刻的近段动作差。诊断固定噪声不等于永久关闭线上随机性。先分析实际执行区间，再看H50远段，不能把远段误差混作当前动作跳变。
4. **核对图像几何和时间。** train视频已为224×224；一集三路抽查均有上下近黑补边，与640×480等比缩放后224×168内容加上下各28像素相符，见 [训练帧](reports/training/redesign-20260916/training-views.png)。不能仅据640×480断言分辨率不匹配，也不能因同D405/kit跳过视角、RGB、左右顺序、曝光、物体像素大小和状态同步核对。30fps下H50约1.67秒；YAM记录约每10步重规划、回复时已过4–5步，本轮不改执行路径。action第0步物理语义仍须以标签和时间戳确认。

### YAM控制优化对成功率的影响：用户追加的只读审查

用户在确认物体不同后明确要求深入分析新增YAM底层控制优化。本次据此只读查看本地YAM控制源码及提交，运行纯数组重放；没有构造StationIO、SDK、执行线程或硬件，没有改动YAM/3588/Thor。YAM观察HEAD为13e5afd；关键变更为a678f3a（100Hz轨迹）与fd966c0（最早有效夹爪预测）。受审文件哈希、计算口径与结果见 [控制审计](reports/training/redesign-20260916/lego3-145144-control-audit.json)。manifest未绑定控制代码/权重版本；下面的逐值匹配支持选择算法一致，不冒充部署版本指纹。

**保留的收益。** 异步worker避免推理阻塞控制；按观测时钟裁掉过期前缀、单个请求在途、过期保持与单写入者机制有明确用途。100Hz二阶轨迹抑制高频跳变；14:51夹爪持续闭合也确有证据。100Hz是轨迹执行频率，模型动作/示范仍30Hz，并不创造新的100Hz视觉决策。

**高优先级问题一：关节和夹爪采用不同预测来源。** `ActionBuffer.current`对三块同目标时刻的12关节近似等权平均（decay0.01），两夹爪直接用最早仍有效的块。直接调用受审ActionBuffer重放全部1370行，1364个有效行与记录policy_action的14维最大差为0：此机制确实解释最新选中目标。夹爪预测所依据的观测年龄中位1.024秒/p95 1.191秒，最新块为0.322/0.486秒；额外年龄中位0.700秒，夹爪通常落在旧H50约第30步，而日志action_index/request只标记最新块。**这些是预测来源年龄，不是网络额外延迟；各候选预测对应的目标时刻仍相同。** 新旧计划已改变抓取点/高度时，旧夹爪的闭合不一定适合混合后的手臂；左/右选中夹爪与最新预测绝对差p95约0.415/0.397，存在旧预测<0.3而最新>0.7的行。该机制解释为何能闭合，也带来动作阶段不一致的风险；尚未证明它是每次抓空的原因。

**问题二：平滑会改变响应。** `TrajectoryFilter`关节使用omega=10 rad/s的二阶临界阻尼、速度3 rad/s/加速度30 rad/s²上限；TrajectoryExecutor的夹爪绕过这一级二阶滤波，但仍受上游反馈相对限幅。隔离该纯数组滤波器，100Hz下0.1 rad阶跃达到50%/90%/95%目标分别为0.16/0.40/0.49秒。这是受控滤波测试，不是真机延迟；不能与预测年龄、网络延迟简单相加。`core.py`和`run.py`的关节每tick反馈相对包络约±0.1 rad、夹爪±0.0333，亦会改变跟随。关节受到多级处理、夹爪受到不同处理，闭合和下降/抬升可能失配；“运动更顺”不能代替抓取验收。

**问题三：采集与部署动态并不相同。** 当前人工来源默认无应用层关节/夹爪slew，100Hz轨迹只作用于policy来源。现场专家动作需包含到位、闭合、稳定抬升，避免教学动作与部署能实现的跟随动态明显不同。先验收这项差异，再讨论是否统一处理路径；不能盲目让模型用更多数据抵消仍在改变的控制器。原始数据同时保留观测时刻状态、原始人工目标和实际提交；正式监督按经审查的专家提交目标语义导出，不能以model policy动作代替专家。

**问题四：当前记录不足以精确归因各级延迟。** 高速路径 `StationIO.apply`提交新的目标后读executor.latest，HDF5的submitted_action是最近一次高频提交快照；submitted_at是在主循环读取时记的时间，并非每次100Hz电机写入时戳。`constraint_mask`定义为submitted与selected不同，滤波/线程相位本身也会使它为真，不等同真正触碰限位。14:51关节每帧最大policy→submitted差p95 0.341 rad，submitted→feedback差p95 0.0647 rad，包含不同时间/滤波/跟随因素，不能直接换算末端毫米误差。下一次诊断应记录各级目标及各自时间、滤波速度状态、夹爪来源request/model index，保留三路RGB与闭合事件。

**验收顺序。** 固定100000和初始布局，先人工对同16 mm小颗粒做10次抓稳抬升；再在其他条件尽量相同下交替做当前小颗粒/接近ABC尺度物体各10次，分别记录抓稳、放置和失败类型。物体变大同时改变视觉与接触几何，只能作为诊断，不能据此独立定位相机问题。随后针对同一物体，一次只改变融合、夹爪来源或关节滤波响应之一；先做离线有界轨迹/事件检查，再由YAM侧安排真机小批验证，保持既有速度/加速度/故障边界。不把直接移除所有平滑、固定延迟关爪、始终强制闭合或更长盲执行列为默认修复。以闭合前后的对位/高度/抬升结果和成功率选方案，不只优化反转次数。

当前manifest RTC=false。普通异步+时间平均不是RTC；PI原作者也指出对不一致动作块直接平滑不保证有效动作，RTC是在生成时约束已承诺动作前缀。若单因素检查仍显示块间不一致，可另做RTC实验，需审计当前TensorRT路径是否支持，并保留关闭/回退；本轮没有开启。来源：[PI RTC说明](https://www.pi.website/research/real_time_chunking)、[原论文](https://arxiv.org/abs/2506.07339)。

### 现场16 mm小颗粒的补采与适配计划

**当前选择：优先现场适配，不盲续旧全量长训。** 100000训练量仍不足一遍，不代表能力已经充分；但更多原尺度ABC并不能保证解决16 mm物体差异，也不能修复控制相位。保留100000作为起点和回退；如果相近ABC尺度下也完全抓不到，先完成控制/合同及原任务拟合诊断，再决定旧数据继续训练。Evo可利用采集窗口做独立训练试验，现场诊断仍固定Pi基线及控制版本。

首批建议**60条：50条训练、10条独立验证**。这是预算受控的试验，不是成功率承诺；以成功抓取机会和状态覆盖计量，不只看episode或小时。先统一“完整接近→抓稳→抬升→按颜色放置”的采集单元，最后还要验证原来的多积木完整任务。若每条有效动作30–60秒，则60条约30–60分钟有效录像，摆放/检查/重录另计；较长完整分拣回合按实际时长重估。

| 50条训练数据的建议构成 | 采集重点 |
|---|---|
| 30条正常成功示范 | 两臂大致均衡；工作区不同位置、方向和颜色；包含完整抓放，不只录最后关爪 |
| 10条当前困难条件的成功示范 | 当前常夹偏的位置/旋转角度、相邻小块；保持场景可观测，先避免同时扩增所有变化 |
| 10条人工成功纠正 | 从实际偏位、第一次夹空后的状态接管，完成重新对位和抓放；失败policy段保留诊断，不当专家标签 |

验证10条按独立布局/采集段隔离，不能随机拆同条视频的帧；现场候选另用固定20次机器人试验，分别报告抓稳并抬升、正确分色放置、完整任务成功。源模型与适配模型使用同一控制版本。50条有清晰收益时，再针对剩余失败扩到约100条训练数据；无收益则暂停加量，回查控制、标签、图像和过拟合。不是每次失败都要求成倍追加同一种数据。针对策略会访问的偏差状态收集专家纠正，是受DAgger思想启发的采集策略，不宣称已经实现完整DAgger训练。[DAgger](https://proceedings.mlr.press/v15/ross11a.html)和[数据覆盖研究](https://www.roboticsproceedings.org/rss20/p013.html)支持考虑访问状态/变化因素；不提供YAM的固定样本保证。OpenPI自定义DROID教程使用30条示范说明流程，同样不是30条一定足够的任务结果：[官方教程](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md#fine-tuning-on-custom-droid-datasets)。

**具体训练定义：100000权重初始化的新现场SFT run。** 学习内容继承，新增步数从0计量并记录parent_checkpoint=100000；不使用原run的`--resume`直接换数据。实现入口沿用OpenPI训练器：`CheckpointWeightLoader(.../100000/params)`、`resume=False`、独立exp_name/输出目录，创建新优化器/调度；原始JAX权重作为起点，不能用Thor TensorRT引擎做训练。源码依据 `src/openpi/training/weight_loaders.py` 与 `scripts/train.py::init_train_state`。当前`train_lego_full.py`硬绑定ABC全量和base初始化，只支持其原实验的续训，尚无已验收的现场适配入口，不能给它一个新run_name就冒充上述方案。

| 项目 | 首轮候选设置，均未开训 |
|---|---|
| 更新方式 | 继续Pi0.5全参数SFT；4×4090、global batch32/FSDP4，不同时改变为LoRA/Evo |
| 初始化/状态 | 100000 params；新Adam状态，eps1e-6沿用当前稳定候选；该值不证明旧NaN根因已修复 |
| 数据混合 | 每个训练起点约50%现场、50%经过筛选的ABC回放；先保留相近抓取/放置知识，不要求每轮遍历全80/96小时 |
| 数据实现 | 当前loader无已验收的按来源加权接口。可新建等近似帧量的现场+ABC发布集取得约1:1自然采样，或补可恢复的来源采样器；记录真实比例/manifest，不能宣称已支持配置一项即生效 |
| norm/动作 | 先审计现场state及Pi delta/H50统计对旧norm的范围覆盖；首轮候选保持100000配套norm以维持权重坐标。重算现场统计用于审计，若要切新norm则作为独立迁移方案验收，不能静默替换。14D、30fps、关节delta/夹爪absolute保持当前合同 |
| LR/预算 | 候选peak3e-6、warmup200、在5k预算内衰减至3e-7；这是需验证的保守起点，未证明最佳。先在1k/2k/5k评估，5k约3.2小时墙钟（四卡合计约12.8 GPU小时），以近期2.3秒/步估算 |
| 停止/扩展 | 全参数有限/资产绑定为前置；训练与现场验证无收益先停止诊断，任务收益清楚才考虑到10k或补更多数据；不用训练loss单独决定 |

混合比例和文件大小是两回事：1小时现场直接并入96.06小时旧数据均匀按帧采样，仅约1.03%起点来自现场。目标域50%采样可提高关注度，无需凑齐旧数据时长。现场等效暴露量为 `steps × global_batch × site_probability / site_frames`；例如50条×45秒×30fps=67500帧，50%现场采样时2k/5k/10k步约为0.47/1.19/2.37遍现场帧起点。H50标签重叠不乘进独立覆盖率，重复采样不创造新的场景；过拟合仍需用独立布局检查。

采集前先用少量专家样本验收原始记录与显式离线转换：YAM导出合同为相机时刻Follower state和实际提交Follower的absolute action；`--expert-only`筛掉policy/hold/无效观测并按真实缺口/epoch切片，不能跨过滤缺口拼H50。导出机制说明见YAM `docs/convert.md`，还需用现场产物逐值验收。新录制spool提交在14:51记录之后；其出现不证明该次queue full已经解决，正式批量采集前检查完整多集落盘和解码。

### 原数据受控对照：保留全量，建立两个可复现的小集合

以下为最初提出的原数据/模型对照方案，当前现场适配优先。未实际筛选episode或生成训练集；不删除、裁帧或覆盖原始数据。

| 集合 | 初始规模建议 | 用途与选择依据 |
|---|---|---|
| D0链路诊断 | 10–20条成功的完整示范 | 相似初始位姿/物体布局，覆盖接近、闭合、抬升、放置；检查能否学会少量轨迹，训练集拟合不冒充泛化 |
| D1首轮对照 | 约2小时，预计约100条；以实际长度为准 | 从原train选清晰成功示范，覆盖两臂、颜色、位置、抓取方式、不同采集段；不取文件前N集或随机孤立帧 |
| 固定验证及任务集 | 原val中未训练的完整episode，另定20次现场布局 | val用于开发评估；最终选模后用新布局/独立批次复验。近重复或同次连续采集按来源组隔离 |

先保留完整时间序列和所有动作阶段，不随意降fps/截去等待段。检查静止时间、闭合/接触/抬升占比、episode长度及成功标签；有静止段主导梯度的证据后再做独立阶段采样消融，保留接触进出与合理等待。D1→约8小时→全量按收益扩展，不预设越少越好。

subset manifest记录源repo/revision、source episode ID、split、理由、长度、任务和视频/数值哈希。Pi新实验从base开始时按D1 train及Pi delta/H50计算新norm；Evo用原生processor及对应统计。不能把旧100000权重与任意新norm拼接冒充续训。若做100000小集再适配，单独设计norm兼容和新优化器/调度实验，不与base对照混合。

可复用 `train_episodes`、`audit_yam_subset.py`、`convert_yam_subset.py --episode-ids`，须区分源ID与转换后索引。`compute_yam_norm_stats.py`目前针对完整发布train逐集布局，不宣称已支持任意selection manifest；先适配或发布新目录再统计。正式 `train_lego_full.py` 硬绑定全量目录，不直接改它承载小集实验。

### 执行顺序与资源上限

| 顺序 | 工作 | 初始预算与结束条件 |
|---|---|---|
| P0 | 固定100000；大小对照、专家基线和控制时序单因素验收 | 优先1个工作日；先补齐最新控制/录制证据，GPU前向仅在Thor/获准服务器；本地不跑训练 |
| P1 | 现场50条train+10条val，先验收少量转换与专家标签 | 记录完整抓放、困难布局、人工纠正；建立固定20次机器人评估；当前尚未采集 |
| P2 | 100000→现场SFT新run，约50%现场/50%ABC | 1k/2k/5k检查，初始上限5k；新优化器/短调度/配套norm审计，详见上节 |
| P3 | 根据结果扩到100条或进行D0/D1成本对照；Evo准备可提前到P1采集窗口 | 无收益先查合同/控制。原数据D0可1k→3k；Pi base→D1可2k/5k/10k检查。Evo先通过YAM/GPU训练验收，两个stage预算见下节；Thor接入是机器人评估前置，不阻塞服务器训练准备 |
| P4 | 选路线、扩充数据 | 以同场景任务结果和取得结果的GPU小时决策；现场有效后再扩场景，暂不做新全量长训 |

同4卡资源串行安排实验，不假定Pi和Evo能同时占满四卡。P2/P3同时报告样本暴露量与GPU小时，Evo两个stage成本都计入、batch区分每卡与全局；不以相同步数当公平算力对照，不直接比较两种loss。以后比较模型需统一现场+ABC数据、验证和执行条件；仅用原D1训练的Evo与现场适配Pi不构成模型受控对照。旧100000为全量数据训练的业务基线，不是与D1 Evo受控的数据消融。

每个checkpoint先检查全部参数有限和资产绑定，再评估固定验证集：训练/验证loss、12关节逐维与近段误差、两夹爪闭合事件/时序误差、同目标时刻块间不一致。BC误差用于诊断，多解动作下不单独决定成功。任务评估记录接近、抓稳并抬升、正确放置、整任务成功和干预，固定超时/初始条件，每候选先20次，报告x/20及区间；小差异独立批次复验。建议16/20作为进入扩场景的初筛门槛，开测前由任务负责人确定；不能把20次当稳健80%成功率证明。现有3–6秒短运动记录不代替完整任务评估。

随着用户确认16 mm物体差异，现场建议为“Pi0.5保留基线，先验收控制时序，再做100000上的现场小数据适配”；随后根据用户提出的采集窗口，将Evo小预算试验提前并行。未自动切模型、回退Pi LoRA、开训或同步服务器/Thor代码。

### 模型选择依据

OpenPI官方将全量微调列为>70GB显存量级；本项目FSDP4已实现4×4090运行，但4090没有NVLink，96GB分散显存不等于单卡96GB，通信瓶颈是否主导仍需profile。历史batch64短测约3.39秒/步、18.9起点/秒；batch32近期约2.3秒/步、14起点/秒，只看秒/步会误判吞吐。两次环境/稳定性不同，不能据此直接恢复batch64。[OpenPI](https://github.com/Physical-Intelligence/openpi)、[NVIDIA规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)。

Evo-1论文约0.77B参数、VLM初始化加两阶段动作学习，不依赖大规模机器人预训练；Pi本项目实测约3.35B。Evo值得测试成本，但参数比例不保证速度比例，论文结果不能当YAM乐高成绩。“微调Evo”须明确起点：只加载InternVL属于新机器人动作学习，模拟器动作权重不是已会YAM的基座。[Evo-1论文](https://arxiv.org/abs/2511.04555)、[原作者代码](https://github.com/MINT-SJTU/Evo-1)。

Evo专用环境已通过CPU验收，并完成FlashAttention GPU kernel/dispatch smoke及完整基座 VLM 权重加载；完整Evo policy前向反向、YAM、训练和Thor仍未接通。复用LeRobot原生trainer/processors；当前官方说明stage1冻结VLM，stage2加载stage1后新建优化器/调度并应用阶段默认规则。默认448图像的计算量须实测，将224源视频放大不增加细节。接入范围与版本差异归 [10](10_vla_platform.md#2026-09-16--evo-1对照实验的范围)，FlashAttention环境证据见 [报告](reports/environments/evo1-flash-attn-20260916/README.md)，模型权重交接证据见 [报告](reports/environments/evo1-model-transfer-20260916/README.md)，来源为 [LeRobot Evo-1](https://huggingface.co/docs/lerobot/evo1)。

### 采集期间的Evo-1四卡试验

**2026-09-16最新用户意图与推荐，尚未执行。** 可以在现场采集期间用服务器4×4090准备并训练Evo，现场继续以Thor上的Pi100000评估控制和数据。新数据齐备后先给Pi现场适配安排四卡时段，Evo保留完整checkpoint接续；不要让两次训练各自占满同一组卡。当前问题同时有物体差异和控制时序混杂，Evo更小只提供成本试验机会，不保证抓16 mm积木更准。

起点采用固定LeRobot原生Evo实现和预训练 `OpenGVLab/InternVL3-1B-hf`（revision `014c0583a0d4bedf29fbe2dbff4f865eb998e171`，权重已交接到服务器），新初始化机器人动作部分，走stage1→stage2；不继承Pi100000，也不把LIBERO 7D策略改成14D即视为YAM预训练。数据先用上述D1约2小时成功完整示范，约100条只是长度估计，另留独立episode/采集组验证。精选数据覆盖两臂、闭合/抬升/放置及场景变化，保持原fps与动作时间关系。D1模型用于先学YAM分拣动作，仍需新物体现场适配。

下面是**容量验收后的候选起步配置**，均为建议，不是4090实测速率或收敛保证。先用DDP，每卡一份模型；不直接照搬Pi的FSDP4。主训练参数FP32、计算使用BF16 AMP；stage1冻结VLM可按BF16加载，stage2解冻后显式以FP32加载VLM并启用原生梯度检查点。

| 项目 | Stage 1：动作学习 | Stage 2：全模型适配 |
|---|---|---|
| 可训练部分 | 动作头及动作相关投影；冻结视觉语言骨干 | 解冻视觉、语言和动作部分 |
| 每卡microbatch候选 | 4 | 2；若不够则1 |
| 梯度累积次数 | 2 | 4；microbatch为1时用8 |
| 有效全局batch | 4卡×4×2=32 | 4卡×2×4=32，或4卡×1×8=32 |
| 学习率 | `1e-5` | `1e-5` |
| 初始优化器更新预算 | 3k检查，首轮最多5k | 5k检查，首轮最多10k |
| 阶段交接 | 保存完整原生policy、processor和训练状态 | 加载stage1 policy，新run、新优化器/调度，应用stage2冻结规则 |

AdamW候选为weight decay `1e-3`、梯度裁剪1、dropout 0.2；两阶段各warmup约300次优化器更新，再按该阶段预算余弦衰减。LR/正则参考官方LIBERO配方，batch和短预算是本项目成本探索；官方LeRobot参考stage1为5k、stage2为80k，原作者仓库另有20k/80k配方，不能拼成统一官方步数。第一轮短训仍可能欠拟合；验证有收益但未收敛时再决定增加更新数或D1→8小时，不以“10k还不行”直接否定模型。[LeRobot参考](https://huggingface.co/docs/lerobot/evo1)、[作者配方](https://github.com/MINT-SJTU/Evo-1)。

**步数必须换算。** 已安装固定trainer的循环`step`按每次microbatch递增，配置`steps`、保存频率、scheduler/warmup均按该口径；优化器只在累积同步时更新。若stage1累积2，5k次优化器更新需`steps=10000`、300次更新warmup需600；stage2累积4，10k次更新需`steps=40000`、warmup需1200。检查点名和看板须同时记录microstep/optimizer update，不能直接与Pi100k步相比。实现依据与固定源码见 [10](10_vla_platform.md#四卡训练落地前的固定版本检查)。假设D1实际2小时×30fps=216000帧，global32下5k+10k更新约480000个起点，即2.22遍等效样本量；这不是独立覆盖率，H50不再乘50。

训练启动前，在获准服务器计算节点分别做两个stage的短容量与吞吐验收：包括首个Adam状态分配、至少100–200个microstep的稳定区间、真实三相机解码/前反向、保存重载和数值有限性。记录显存allocated/reserved峰值、样本/秒、数据等待及GPU小时；不要用Pi的2.3秒/步预测Evo耗时。先把采集窗口作为首轮墙钟预算，由实测确定能完成多少更新，来不及完成两个stage就保留完整checkpoint。单卡microbatch1仍不够时才另审分片方案；降低分辨率、删相机或冻结更多层都会改变实验，需要单独命名。

**现场数据到齐后的衔接：** 得到Evo的YAM stage2策略后，以同一批50train+10val现场数据进行独立现场SFT；候选用约50%现场/50%精选ABC、较低LR `3e-6`、1k/2k/5k次优化器更新检查，是否延长由独立布局和完整任务决定。新run从Evo权重初始化，保留其动作表示和独立processor合同；新数据统计如何绑定需先验收，原生warm-start会注入当前数据集stats，不能声称旧norm自动保留。若stage2尚未获得有效YAM策略，可在确认stage1动作学习有效后，以混合数据进入stage2，但要标记为不同实验。两模型共用现场划分、20次任务评估和固定控制版本；Pi旧100k是业务基线，两者训练历史不同，不宣称严格预训练算力对照。

## Loss日志与看板口径（2026-09-08用户确认）

- 正式训练保持`log_interval=10`，不改成逐步记录。每一步先对batch、动作时间窗口和模型动作维度
  求平均得到该步loss；日志再对最近10步的loss取算术平均，首条记录例外，仅包含第1步。
- 当前JSONL使用已完成步数：step1对应第1次更新的loss；step11对应第2–11次更新的平均，
  step21对应第12–21次更新的平均，以此类推。这是训练loss，不是关节物理误差或任务失败率。
- 看板“原始曲线”指未经页面平滑的**日志均值**，不是逐步原始loss；平滑线是在这些日志均值上
  再做指数平滑，仅影响显示，与模型权重EMA无关。看板10秒刷新不改变日志统计窗口。
- 用户讨论论文中密集原始点形成的“厚毛边”后，明确决定无需调整记录方式，只需说明上述口径。
  不人为添加噪声，不把线条厚度自动解释为置信区间，不尝试还原已被平均掉的历史逐步波动。
- 依据：`src/openpi/models/pi0.py::compute_loss`、`scripts/train.py::train_step/main`及
  `scripts/training_dashboard.html`的绘图逻辑；运行版本f93a792与启动证据见本页下文。

## 2026-09-09 故障恢复与当前授权

**17:10最新授权与启动：**用户转达管理员允许直接开训、遇问题再处理，并明确要求现在开始，取代下方17:04暂不放行结论；ECC不再作为本轮启动前置条件，根因未解决的事实保留。已核对2064有效、四卡空闲、无训练/诊断进程，r2无完整checkpoint；保留旧现场，从base新建`lego_full_b64_r3_20260909`。17:10:35通过tmux `yam-lego-full-r3`派发Slurm2064.81，PID1061106，沿用不可变a53bb00快照，batch64/FSDP4/40k/每5k保存、LR/精度/数据均不变。看板切到独立r3缓存。启动验证见[r3记录](reports/training/pi05-r3-20260909/README.md)。

**17:04用户要求先排查再训练：**正式训练暂不放行。本轮全量4458个Parquet/10,374,181帧哈希与norm来源一致，14D state/action全部有限，19项轻量合同测试通过；未运行训练或训练smoke。17:03四卡空闲，CPU CE137754、UE0，持续增长；根因仍未知，不能归咎数据/算法或反向断言硬件。当前小时巡检为PAUSED。范围和恢复门槛见[本轮复核](reports/training/pi05-crash-audit-20260909/README.md)。此状态优先于下方历史恢复授权。

**11:42状态更新：**NCCL定向诊断2064.73在10:58:40 TIMEOUT，进度条到6步，约130秒/步；
日志没有捕获非法访问或Internal Sanitizer Error，也无最终ERROR SUMMARY，不能记为通过。
正式r2仍停在271步历史日志，没有完整checkpoint；当前没有训练/诊断进程，四卡空闲于allocation2064。
gpu002仍全部分配；CPU CE131367、UE0。看板服务与镜像正常，仅展示历史值。
两次有限插桩均未给出根因，不重复同类耗时测试或原样重开第三轮；后续优先复核管理员对
CPU0_DIMM_B1和GPU1/Xid13/43的检查、健康替代节点或新的可验证软件修复证据，再恢复训练。

**10:43诊断更新：**2064.71完整memcheck在10:13:40 TIMEOUT，只观察到step1；工具多次提示
无法分配插桩内存、部分kernel未检查，因此既非通过，也不能解释为原训练OOM。CPU CE增至131365、UE0。
启动独立`lego_ncclcheck_20260909`短测（tmux `yam-lego-ncclcheck`）：同快照、20步、15分钟上限，
仅`--kernel-name kns=nccl`检查通信内核，`--force-synchronization-limit 100`限制诊断积压，
XLA显存池从0.92降至0.85给工具留空间；batch64/精度/优化器不变。这不是正式第三轮。
日志位于 `/home/wuyan/lyj/YAM/training-runs/control/lego_ncclcheck_20260909/diagnostic.log`。
该过滤诊断即便通过也不能排除未插桩XLA计算kernel/硬件或长时问题；不要直接据此宣称全量训练恢复。

**09:41再次核验：r2已于09:33:21崩溃，下面09:20启动成功不代表当前在训。**
最后logged step271，进度条278，SIGSEGV；NCCL报告illegal memory access，内核GPU1（PCI52:00.0）
Xid13 Out Of Range Address，继发Xid43。CPU corrected ECC计数127141，UE0，不能直接归因硬件。
gpu002四卡均已分配，无空闲替代四卡节点，不取消其他作业。原样重启已复现，不再盲重开正式run。
09:43:27启动独立诊断 `lego_full_memcheck_20260909`、tmux `yam-lego-memcheck`，同a53bb00快照，
使用`/usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 86`包裹正式入口，
batch64/FSDP4不变、仅20步、Slurm限时30分钟，NCCL_DEBUG=INFO；不是40k正式训练或已修复。
诊断日志 `/home/wuyan/lyj/YAM/training-runs/control/lego_full_memcheck_20260909/diagnostic.log`。
下一次巡检先查诊断结果/实际步骤与退出码，再决定有证据支持的代码或环境修复，不重复启动诊断。
插桩额外显存/超时不能当原训练OOM或训练健康验收；需管理员排查时提供GPU1 Xid13/43和DIMM_B1证据。

以下取代旧运行的保存周期与“无checkpoint等待确认”边界：用户已授权主动排障、修复并恢复训练，
保存/保留间隔均为5000步；batch64/FSDP4、LR、精度、数据和40k阶段终点不变。
有完整checkpoint优先真续训；没有则允许保留旧现场，在新run从base重开，不能混接旧步数或loss。
反复原样重试不构成修复，驱动/硬件需管理员权限时提供证据并寻求安全替代，不擅自升级系统驱动。

9月9日09:06现场核验：旧步骤2064.33于05:21:05以SIGSEGV退出，最后日志16031步，loss0.01768966，
旧run仅metrics、没有正式checkpoint。四卡allocation仍有效、GPU空闲。core虽然journal访问受限，
其文件ACL允许本用户读取，已解压并通过GDB检查；PC `0x7fee0e2787c7` 落在
`/usr/lib64/libcuda.so.595.45.04` 映射内。core截断为1GiB，栈页缺失，不能据此宣称已定位最终根因。
EDAC累计CE127134、UE0，并有CPU0_DIMM_B1单bit corrected ECC记录；未建立与05:21退出的因果关系。
无已见OOM/磁盘满证据。现场证据与管理员建议见
[故障记录](reports/training/pi05-recovery-20260909/README.md)。

本次恢复目标run为 `lego_full_b64_r2_20260909`，与旧run隔离。入口新增 `LEGO_RUN_NAME`，
原生崩溃启用 `PYTHONFAULTHANDLER=1`，`--resume` 无已提交checkpoint时强制拒绝。
看板service同步新run和独立缓存 `artifacts/training_dashboard/recovery_20260909/metrics.jsonl`，
显式传入save-interval5000；每10秒刷新不调用模型。每小时巡检`pi0-5`已更新为主动恢复模式。
9月9日09:16:39已提交新进程：Slurm2064.63、PID3091182、tmux `yam-lego-full-r2`，
固定Gitea快照 `/home/wuyan/lyj/YAM/env-transfer/lego-full-a53bb00`（a53bb001e4acb2cce486f2da83d6d8256439a188）。
启动命令：`LEGO_RUN_NAME=lego_full_b64_r2_20260909 bash scripts/launch_lego_full.sh 2064`。
恢复时使用同一快照/环境/run，加`--resume`，先核验完整checkpoint与无重复进程。
实际`initial_config.json`已核对save_interval=keep_period=5000、batch64、steps40000、resume=false。
新控制目录 `/home/wuyan/lyj/YAM/training-runs/control/lego_full_b64_r2_20260909`，
新run目录 `/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64_r2_20260909`。
09:20启动验收：已核对真实日志21→31→41，step41 loss0.04490658、grad_norm0.28559217、
3.364秒/步；四GPU100%、约22828MiB/卡、56–61°C。看板API已同步41步、save_interval5000、
无同步错误。此为启动验收，首份完整checkpoint与长期稳定性仍须后续验证。

## 故障诊断复用与重试条件

此节整理 2026-09-09 的历史诊断，不报告当前进程或改变后续授权。当前节点、驱动状态和首个非法读写算子均需新证据；本轮记忆维护未查询服务器。

| 问题与原因假设 | 实际尝试与结果 | 判断、局限与重试条件 |
|---|---|---|
| r2 出现 Xid13/43、NCCL illegal memory access；完整插桩能定位首个非法访问 | 同 a53bb00 快照、batch64/FSDP4，20步/30分钟 memcheck；只观察 step1，插桩内存不足、部分 kernel 未查，TIMEOUT | inconclusive；工具内存不足不是原训练 OOM。只有可用插桩资源/时限、缩小且能触发问题的案例或诊断策略发生变化才值得重做；新案例须另存证据 |
| 通信内核可能是非法访问来源 | NCCL-only 20步/15分钟；同步积压限100、XLA池0.85；到6步 TIMEOUT，无最终 ERROR SUMMARY | inconclusive；过滤掉计算 kernel 且改变池配置，未捕获不等于排除通信/硬件。得到新的通信线索或足够覆盖的诊断窗口后再重试，不原样重复耗时测试 |
| 数据损坏/非有限输入可能导致失败 | [全量复核](reports/training/pi05-crash-audit-20260909/README.md)：4458个Parquet/10,374,181帧哈希匹配，14D state/action有限，19项轻量测试通过 | 已检查范围未见异常；未全量解码视频、未复验基础权重完整哈希，也不证明GPU算法正确。数据/变换/权重版本变化或有具体异常样本时重查对应部分 |

前两项实际参数、远端日志路径与退出观察见 [故障恢复记录](#2026-09-09-故障恢复与当前授权)；本地报告只保存其结果摘要，完整诊断日志本轮未重新取回，不能补写未知错误栈或声称根因已定位。CPU CE增加与GPU错误的因果关系仍未知。恢复训练是授权下的操作，不是修复有效性的证明。

复用检查时先区分：配置要求 batch/FSDP/保存周期、`initial_config.json` 的启动记录、实际日志步数/退出码，以及完整 checkpoint 的独立进程恢复结果。配置为每5k保存不证明文件已完整提交；看板有曲线不证明进程还活着。历史命令中的 Slurm ID、快照和阶段终点不能直接作为当前运行参数。本地仅运行轻量合同检查，训练和插桩执行须在获准计算资源上。

## 正式运行：2026-09-08 Lego全量微调（历史启动记录）

用户已授权并于北京时间14:03启动，首个更新14:05:36完成。运行在Slurm2064的步骤2064.33、
gpu001、tmux `yam-lego-full`；这些是启动观察，巡检必须重新查询。固定代码快照
`/home/wuyan/lyj/YAM/env-transfer/lego-full-f93a792`（提交f93a792），从Gitea获取。
batch64/FSDP4/EMA关闭、40k阶段终点、每20k完整保存，LR周期162097，详见
[启动证据](reports/training/pi05-full-launch-20260908/README.md)与其中`initial_config.json`。

- 正式run：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64`。
- 控制日志：`/home/wuyan/lyj/YAM/training-runs/control/lego_full_b64/train.log`；配置和恢复测试证据同目录。
- 启动器：固定快照的`scripts/launch_lego_full.sh 2064`，通过tmux启动；锁保护避免重复launcher。
- 仅在无活跃训练、完整checkpoint和当前资源已核验后，使用同一快照启动器加`--resume`。
  `--steps`表示累计停止点，不是额外步数；不得自动超过40k。不要使用overwrite或base权重冒充续训。
- 全尺寸Pi0.5 checkpoint恢复30→31、GPU新进程训练入口恢复2→4并再次保存均已通过。
  CPU同进程测试在第二次JAX调用Aborted仍未定位，不能写成已修复；正式运行/恢复使用GPU独立进程。
- 已核对远端与看板步数1→11→21→31→41，loss和梯度有限，近期约3.37秒/步；
  瞬时四卡利用率100%。只是启动验收，不是训练完成或任务效果验收。
  第一份正式checkpoint须到20k才产生，此前只有恢复测试产物，不能混淆。

## 2026-09-08 四张 4090 全参数短测

`pi05_yam` 在Slurm2064/gpu001的4×4090上通过全参数容量测试，33.53亿参数全部可训练。
先分片加载checkpoint修复初始化OOM；随后预分配显存消除了batch32按需增长时的失败。
固定FSDP4、EMA=None、原AdamW、三相机/文本200/H50/32D、JAX92%显存预算：
global batch92通过3步、相邻96与128 OOM；这不是改变精度/卸载/预算后的硬件绝对上限。
推荐batch64作为长训候选：固定batch10步通过，活跃峰值18.54GiB，3.3385秒/步；
全部训练集随机取数、8worker的30步测试平均3.3895秒/步（去掉首步），平均取数等待0.0328秒。
这是短测，不保证长期稳定或任务收敛；正式长训未启动，默认LoRA配置仍保留。
用户随后确定阶段计划：batch64、原AdamW峰值2.5e-5、warmup1000、约162097步余弦至2.5e-6，
首阶段累计40k，每20k完整保存，后续评估后续训向约一遍数据推进。不是每40k重置学习率。
按3.3895秒/步，40k纯训练37.7小时（排期40–48小时），一遍约152.6小时，不含停机/评估。
此计划替代短测报告中的最初30k/60k方案；正式运行见本页顶部，验证/早停尚未接入循环。
限定条件、保存证据、参数/步数解释与时间外推见[batch测试报告](reports/training/pi05-batch-limit-20260908/README.md)；
初始化修复及最初batch4证据见[首轮报告](reports/training/pi05-full-20260908/README.md)。
全量train-only norm见[数据合同](04_data_contracts.md#2026-09-08-全量发布与归一化完成)。

## 训练看板与每小时巡检（2026-09-08）

- 页面：[training_dashboard.html](../scripts/training_dashboard.html)，本地入口 `http://127.0.0.1:8765/`。
  W&B风格的只读工作台，不使用W&B上传或外部CDN。主图loss，附验证loss、LR、梯度/参数范数、
  单步耗时、吞吐、显存、GPU利用率、取数等待；日志未提供的指标明确留空。
- 服务：[training_dashboard.py](../scripts/training_dashboard.py)，仅Python标准库、仅监听loopback。
  10秒SSH拉取指定JSONL并原子替换本地缓存，页面10秒轮询；两者均不调用模型，不消耗模型token。
  SSH失败保留上次数据并提示；忽略未完成的尾行、报告坏行、处理step回退，不读取/修改checkpoint。
  单文件上限32MiB，最多展示最近20000条日志记录，超限明确提示；这不是训练步数上限。
- 本机用户服务 `training_dashboard.service` 已启用，文件位于 `scripts/`；随用户服务管理器启动，
  异常退出10秒重启。查看/恢复：`systemctl --user status training_dashboard.service` /
  `systemctl --user restart training_dashboard.service`。本机休眠、断网或用户服务未运行时不能保证刷新。
- 本地缓存 `artifacts/training_dashboard/live/metrics.jsonl`；预留远端
  `/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64/metrics/metrics.jsonl`。
  已绑定正式run并核对多次真实步数增长；不能仅凭页面在线断言训练健康。
  参数面板显示讨论计划，不冒充实际配置验真。原生LocalMetricLogger已有loss/grad_norm/param_norm/LR；
  正式入口已补记时间、近期步速/吞吐与JAX活跃分配（不是nvidia-smi显存或峰值）；验证loss和GPU利用率
  仍需独立采集，留空不伪造。10秒刷新不等于每10秒新增loss，log_interval=10时约34秒更新一组。
- 启动例：`python3 scripts/training_dashboard.py --metrics /absolute/run/metrics/metrics.jsonl`；
  远端镜像加 `--remote yam-server --remote-metrics /absolute/remote/metrics.jsonl`。
  服务内的参数与路径通过CLI设置；默认stage40k/total162097/save20k，不启动训练。
- 每小时heartbeat `pi0-5`（“每小时巡检Pi0.5全量训练”）已设ACTIVE，无终止日期；
  只在异常/修复/重要进展时通知。尚未启动训练时不自行首次启动；40k阶段结束不自行越过停止点。
  巡检现场Slurm、计算进程、步数增量、有限loss/梯度、GPU、磁盘/NFS及完整checkpoint。
  故障恢复先排除编译/保存/排队，确认无重复写进程、资源有效、配置/数据一致和恢复链路通过，
  再用该run记录的启动器续训，验证多个新步数；不改batch/LR/精度/预算。同一修复两次失败停止盲重试。
- 用户允许跳过部分经审计证实损坏的样本：保留原件和源ID/原因/哈希/排除清单，优先整轨迹隔离到
  新数据版本、保持三相机/state/action对齐；自动修复单轮最多5条、累计不超过原train的1%，超出请确认。
  不能把OOM、网络、存储或解码依赖故障当作数据损坏。数据集合改变需记录版本分支、验证采样/续训位置，
  按合同重算验收norm，不能宣称原序列无缝续接。无法验证则保留现场报告。无限巡检不等于无限重启。
- 验证：13项pytest覆盖日志追加/半行、非法值、续训回退、读取上限、HTTP路径隔离、镜像失败保留及成功发布；
  JS语法检查、实际HTTP200与等待远端日志API通过。未宣称浏览器交互或正式训练的端到端验收。

本页只描述当前 YAM 训练路线。服务器和环境先看 [02 · 服务器与环境](02_installation_and_environment.md)，动作/图像合同看 [04 · 数据合同](04_data_contracts.md)。

## 历史默认路线（2026-09-07，已被全量微调决定覆盖）

服务器训练关闭 W&B（`wandb_enabled=False` / CLI `--no-wandb-enabled`），不依赖 W&B 在线或离线运行。训练入口已有 `LocalMetricLogger`，将指标写入实验目录的 `metrics/metrics.jsonl`、`metrics/metrics.csv` 和 `metrics/plots/`；结合 Slurm stdout/stderr 和 checkpoint 作为记忆证据。日志存在不等于 checkpoint/部署 gate 通过。

首选配置为 `pi05_yam_lora`：

- Pi0.5，`gemma_2b_lora` + `gemma_300m_lora`；
- 冻结规则由对应 `Pi0Config.get_freeze_filter()` 生成，LoRA 关闭 EMA；
- YAM 双臂真实动作 14D，模型内部 padding 为 32D；
- action horizon 为 50；
- 初始保守默认值为 batch 4、workers 2、每 1000 步保存、保留周期 5000、总步数 30000；这些是新服务器上的起始值，不是 GPU 性能结论。

`pi0_yam`、`pi0_yam_lora`、`pi05_yam`、`pi05_yam_lora` 均已注册在 `src/openpi/training/config.py`。默认 `repo_id=local/yam_bimanual` 只是占位符，正式训练必须用 CLI 或复制配置覆盖为已审计的数据路径。

阶段顺序固定为：第一阶段用已审计的乐高分拣示范做 Pi0.5 LoRA SFT；第二阶段再接入 DAgger，用人工纠正/回放数据建立独立数据版本和实验名。当前仓库只提供 YAM 的输入输出合同和通用训练入口，DAgger 的采集、纠正合并和安全 rollout 尚未宣称完成，不得把普通 SFT 结果写成 DAgger 结果。

## 训练前 gate

按以下顺序执行，任一步失败都不启动长训：

1. 确认当前 commit、数据版本、episode split、config 和初始化 checkpoint。
2. 检查 LeRobot metadata、`action`/`observation.state` 的 14D、三路图像、task/prompt、视频首中尾解码。
3. 按同一训练数据版本计算 norm stats，保存为 `assets/yam/norm_stats.json`。
4. 用真实 dataset loader 取样，确认 transform 后 state/action 能进入模型的 32D spec。
5. 在 Slurm GPU 分配内做短步数 smoke，再启动正式训练。

基础代码测试：

```bash
"$PYTHON" -m pytest --strict-markers -m "not manual" -q
```

## Norm stats

YAM 的 norm 必须在 `YamInputs` 和 delta action transform 后计算；不要复用 OpenArm、Piper 或其他单位合同的 stats。示例：

```bash
DATASET=/home/wuyan/lyj/YAM/YAM_data/audited/yam_lerobot_v001
"$PYTHON" scripts/compute_norm_stats.py pi05_yam_lora \
  --repo-id="$DATASET"
```

实际输出根目录和资产目录以脚本日志为准，完成后确认 `yam/norm_stats.json` 可读且与训练数据版本一致。`compute_norm_stats.py` 不会替代数据结构审计。

YAM 的脚本默认输出已对齐训练读取目录：默认配置写入
`assets/pi05_yam_lora/yam/norm_stats.json`，不是 dataset 根目录。
自定义 `--output-dir` 时必须同时使训练的 assets 配置指向同一父目录。
每个正式数据版本使用独立 assets 目录，避免重算 norm 覆盖另一实验的统计量。

## 单机 smoke 和正式训练

先在已分配 GPU 的节点运行 10～20 步，使用新实验名和独立输出目录：

```bash
CONFIG=pi05_yam_lora
DATASET=/home/wuyan/lyj/YAM/YAM_data/audited/yam_lerobot_v001
EXP_NAME=yam_pi05_lora_smoke

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PYTHON" scripts/train.py "$CONFIG" \
  --data.repo-id="$DATASET" \
  --exp-name="$EXP_NAME" \
  --num-train-steps=20 \
  --batch-size=1 \
  --num-workers=0
```

smoke 通过后再用保守默认值启动正式训练；长任务使用 Slurm/tmux：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PYTHON" scripts/train.py pi05_yam_lora \
  --data.repo-id="$DATASET" \
  --exp-name=yam_pi05_lora_v001 \
  --num-train-steps=30000
```

当前使用 JAX 训练入口；PyTorch LoRA 支持须另行审计和验证，不属于本轮已验收路径。
续训只从完整 checkpoint resume，不能覆盖旧实验。

## 参数和实验隔离

2026-09-07 源码审查：当前继承 CosineDecaySchedule（warmup 1000 步，peak LR `2.5e-5`，
decay 30000 步，终点 `2.5e-6`），AdamW（b1=0.9、b2=0.95、eps=1e-8、weight decay=1e-10、
global gradient clip=1.0）。LoRA 的 PaliGemma rank/alpha=16/16，action expert=32/32。
冻结过滤器冻结对应 LLM 主干并排除 LoRA；不能将其表述为全模型“只有 LoRA 可训练”，
其他未命中过滤器的参数仍可训练。EMA 关闭，W&B 关闭；正式开跑前记录实际 trainable parameter 数和显存。

30 FPS 数据上 horizon 50 对应约 1.67 秒预测窗口，不等于机器人每次必须执行 50 步。
batch 4 × 30000 steps 约采样 120000 个训练窗口；manifest 声明数据约 97.586 小时（含验证集），
因此 30000 步只是初始预算，不是完整 epoch 或已验证的收敛方案。
完成清洗后按实际 train 帧数记录 `steps * global_batch / train_frames`，结合固定 holdout 决定训练长度。
20 步 smoke 位于 warmup 初段，只验证训练链路，不作学习效果结论。

当前首选是 `scripts/train.py` 的 JAX 路径；不要把该 LoRA 配置直接视为 PyTorch 入口已验证支持。
上传子集的转换/发布流程见 [数据合同](04_data_contracts.md#abc-乐高子集上传期间的清洗流程)。

| 参数 | 入口 | 规则 |
|---|---|---|
| 数据路径/split | `--data.repo-id` 或独立配置 | 变化就新建数据版本并重算 norm |
| LoRA/全量 | config 的 model/freeze filter | 首轮优先 `pi05_yam_lora`，不要混用 checkpoint |
| batch/workers | config 或 CLI | 先以 GPU smoke 测定，OOM 后降低 batch/worker |
| horizon/action dim | model config + YAM contract | 当前为 50/32 内部、14D 外部；不能随意改一端 |
| 保存/步数 | config 或 CLI | 输出目录包含 config、实验名和 step |

每个实验至少记录 git commit、config、repo id、数据版本、split、norm 路径、base checkpoint、LoRA 设置、batch、workers、step 和 seed。普通 SFT、不同 LoRA 设置和后续评估必须使用独立实验名，避免结果无法归因。

面向跨会话记忆的产物身份、指纹和本地 run manifest 合同见 [09 · 记忆系统](09_memory_system.md)。该合同用于记录与验收；训练入口尚未自动生成其全部字段，不得把文档规范写成已经实现的采集器。

## Checkpoint gate

训练日志显示完成不代表 checkpoint 可用。部署或评估前检查：

- step 目录的 Orbax 参数元数据完整；
- `assets/yam/norm_stats.json` 存在且与 config/data 绑定；
- config、git commit、数据版本和训练日志可追溯；
- 用 [05 · 训练后 policy smoke](05_inference_and_rollout.md) 验证输出为有限 `(50,14)`；若部署 Thor，还要按 [08 · Thor 端侧部署](08_thor_edge_deployment.md) 保留 JAX reference 与转换后 engine 的验证报告。

半写入数字目录、缺少 norm 或只有单独 `params/` 的目录不得部署。

## 评估与结果记录

离线 loss、动作误差和 chunk 连续性只能作为诊断，不能直接等同于 YAM 真机成功率。固定 holdout 上比较时必须保持数据版本、prompt、horizon 和 norm 一致；真机或服务结论另行记录 checkpoint、版本和 smoke 证据。结果原因写入 `docs/07_change_log.md`，不把历史 OpenArm KAI0 计划混入当前 YAM 结论。

当前评估入口示例：

```bash
"$PYTHON" scripts/evaluate_checkpoint.py \
  --config=pi05_yam_lora \
  --checkpoint-dir=/path/to/checkpoint \
  --dataset="$DATASET" \
  --val-split=80:100
```

该入口只做数据/动作合同、首步误差和 chunk 连续性的离线诊断；DAgger 只有在纠正数据、采集协议和安全 rollout 均单独留痕后才进入第二阶段。
