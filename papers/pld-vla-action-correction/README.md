# PLD 式 VLA 动作修正 · 论文研究工作区

**职责：文献调研、研究设计与论文写作。项目代码、数据处理、训练和部署由其他工作流推进。**

当前处于想法与实验逐步确定的阶段。先把已有方法读清楚，不强求现在提出原创算法。本目录没有 RL 性能结果。

## 建议阅读顺序

1. [文献调研：共同骨架与实质差异](literature-review.md)——先读这个。
2. [PLD 路线研究设计](drafts/research-plan-v0.1.md)——架构、动作、奖励、学习与蒸馏。
3. [实验问题与证据矩阵](experiment-matrix.md)——哪些实验分别回答什么。
4. [先例与候选想法审计](novelty-audit.md)——可选讨论，不是已认定创新。

## 来源与边界

- [公开文献登记](sources/literature.md)：25 条来源，区分方法选读和摘要核查；[BibTeX](sources/references.bib)。
- [14 集历史材料与指标口径](sources/user_materials.md)：分类完成不等于检测准确率。
- [事实底稿](01_research_canon.md)、[主张—证据](02_evidence_table.md)、[候选论证地图](03_argument_map.md)。
- [范围](00_scope.md)、[章节合同](04_section_contracts.md)、[写作规范](05_style_guide.md)、[审阅记录](qa_logs/review.md)、[状态](state.json)。

当前共识：以冻结 Pi0.5＋小型整段动作修正专家为研究起点；奖励模型、探索空间、策略更新位置和能力吸收方式分轴比较。RECAP 不被预设为必须人工逐阶段标注，RLT 不被预设为必须实时人工接管。

后续每次只更新对应主题与证据表，保留来源版本；不为普通进度不断复制新计划。不以本目录研究设想替代已确认的项目训练配方。
