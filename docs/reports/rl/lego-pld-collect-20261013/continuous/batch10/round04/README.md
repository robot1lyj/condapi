# R04：连续轨迹合同下的四种学习器对照

本轮使用Thor，冻结Pi 200000及已有10集、reward-release-v1、learning-replay-r03-v2。没有新交互、没有把24w Pi数据混进来。用户明确选定下一批仍用20w Pi；暂停前的纯24w推理容器保留，不修改其权重、引擎和配置。八组训练及48份checkpoint审计均已完成，生产专家没有随训练结果自动替换。

## 可复核配方

代码`229bc2f72b2b2f54ac15bb2803384d885ca061ff`，Thor训练镜像`sha256:325ef815f1ae6b153c04425ff96d55595c2d17a86ee29b56cbe5752201780afd`。每组3层256宽FP32 MLP，冻结ResNet18，batch256、lr3e-4；先200次训练集行为中心坐标对齐，3000次critic中前1000次MC初始化、critic预热到2000次，最后1000步每2步更新actor一次，共500次。训练集228尝试/825决策，开发验证集E01共17尝试/57决策；按尝试均匀，成功池与全部池各占50%。状态/视觉/计划分组RMS仅由训练集计算。关节±0.08rad、夹爪±0.05，RTC、奖励、可行映射及连续解码器完全相同。

- **PLD-CT-SAC**：显式熵项soft-Q backup、温度学习，无成功BC项。学习器遵循SAC目标；冻结VLA、残差接口与RTC分段为本项目实现，不称逐项复现PLD。
- **PLD-CT-IQL**：expectile=0.7，优势温度3、权重最多100；V拟合记录动作的target Q，Q的下一状态backup不查询actor；对可行残差做优势加权Normal似然，折叠动作轴不计似然。学习用的全局log-std单独保存，不改变线上collect噪声配置。这是IQL适配可行动作的变体。
- **PARTS-CT-TD3BC**：复用R03有约束基线，含成功BC、失败零修正权重0.1；是项目变体。
- **BC-CT-Success**：复用相同训练路径，actor的RL系数为0，作为成功BC与失败零锚定控制；critic只供诊断。

各算法seed0/1。五种合成/CLI GPU验收（另含CalQL）全部通过，包括SAC熵backup、IQL无actor backup、终点、保存恢复、坐标对齐、可行轴与导出。完整训练循环合计273.33秒，另有环境启动、数据处理和检查时间；这是小网络训练，不是Pi全量微调。[数值验收](gpu-checks.json)、[八组结果](results.json)、[启动身份](status.json)。

## 结果与边界

| 方法 | 入口AUC seed0 / 1 | 入口记录回报RMSE seed0 / 1 | 全验证动作超训练边际1–99%比例 seed0 / 1 |
|---|---:|---:|---:|
| BC | 0.800 / 0.783 | 0.456 / 0.467 | 0.75% / 0.25% |
| TD3BC | 0.700 / 0.650 | 0.572 / 0.578 | 0.00% / 0.00% |
| IQL | 0.850 / 0.867 | 0.426 / 0.394 | 1.00% / 0.00% |
| SAC | 0.583 / 0.583 | 0.721 / 0.724 | 9.52% / 10.78% |

常数训练均值的入口RMSE为0.455，按左右臂均值为0.450。IQL两seed入口排序及记录回报诊断优于本轮TD3BC，值得进入后续真机比较；验证只有一个episode，不能把AUC当成功率或宣布泛化。SAC的Q含熵，其MC误差不能和其他算法直接排名；它的动作超范围比例约10%，当前不优先直接试跑。这个范围指标只是各维边际诊断，不是严格的状态条件分布外检测。

四组TD3BC/BC重跑的最终actor文件SHA及整份latest-evaluation与R03完全一致：[重复性核查](reproducibility.json)。本轮新增8个完成运行，其中4个是有意重复基线；不能把它们当8种算法或独立机器人实验。旧33运行的[历史清单](../algorithm-inventory.json)保留其原统计时点。

## 决策

[Closing-Collect r05](../closing-collect-r05/README.md)已完成本机验收并上线，使用已验收Actor-2作为行为中心，保持actor权重和eval物理输出；单独增加闭合方向探索、保持张开与关节范围、平滑噪声、RTC前缀和抓取范围限制。这能分离“探索余量改变”和“换学习器”的效果。新范围未经历真机结果之前，不把旧奖励重新分配给想象的闭合动作，也不把eval输出直接倍增。

IQL保留两个seed与所有检查点，先不挑17条验证中分数最高的seed宣称最优；独立布局的真机对照决定其效果。新经验到位后，同合同重训TD3BC/IQL/SAC/BC，不把不同范围或不同Pi版本混排。第一批需要少量collect核对闭合修正是否确实到达，再决定扩到10集；现有10集继续用于初始化与对照。

## 原件与复现

Thor：`/home/wuyan-lyj/thor/rl-training/runs/pld-ct-*-r04-s*`保存每500步完整optimizer/actor/critic（IQL含V与log-std）及bundle、逐步指标和验证预测；`continuous10-r04-orchestration`保存启动命令、状态和逐checkpoint审计。源码通过Gitea获取，数据只读挂载；原始数据、旧run与24w服务未删除。报告同目录保留每组run身份、launcher和audit，配方归`configs/pld_rl/continuous10-r04/`。已为八组原件272个文件、1.91GB生成逐文件SHA256索引：[archive-index.json](archive-index.json)，包含所有检查点、optimizer、验证预测、指标与bundle；未删除任何旧run。本地8766继续复用VLA看板，缓存`artifacts/pld-continuous10-r04`；历史缓存保留。

方法来源：[PLD](https://arxiv.org/abs/2511.00091)、[SAC官方实现](https://github.com/rail-berkeley/softlearning)、[IQL官方实现](https://github.com/ikostrikov/implicit_q_learning)、[TD3+BC官方实现](https://github.com/sfujim/TD3_BC)。论文名称不代替真机证据，具体工程差异以上述配方为准。
