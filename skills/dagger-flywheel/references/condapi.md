# condapi 项目接入

只在实际仓库根目录匹配该项目时读取。skill保存方法，最新事实全部从现有owner检索；不将用户提供的schema说明视为真实文件已验收。

## 查阅路线

- AGENTS.md与docs/cache/context_index.md：工作区、记忆、运行授权与本地禁止训练边界。
- `docs/03_training_and_evaluation.md` → `docs/reference/lego_dagger_playbook.md`：本项目首轮配方、训练与真机对照。20h只是历史实例；后续campaign可用任意数据规模的首次任务SFT基线。
- `docs/04_data_contracts.md`：YAM14D、三RGB、absolute目标、训练时关节delta/夹爪absolute、HIL字段映射与有效位。
- `docs/02_installation_and_environment.md`：当前可用服务器/Conda/作业入口；运行前复核，不把旧路径与GPU审计当现状。
- `docs/08_thor_edge_deployment.md`及其下属手册：部署验收。需要交接Pi checkpoint时使用已安装的`thor-checkpoint-deploy`，按真实RTC合同选择路线，调用前读取该skill；不读取或操作3588实现。
- `docs/07_change_log.md`：保存本轮决策原因和结果。运行产物放既有`docs/reports/training/`报告或获准服务器实验目录，以索引链接，避免把大数据进Git。

## 现有工具与能力核对

`train_lego_full.py`是Pi正式入口；此前已支持parent权重、新run、RTC目标、子集清单和resume合同，但曾固定单数据源。每次从当前代码核对混采/有效起点功能，不把skill安装当这些功能已经实现。

`scripts/audit_yam_subset.py`和`scripts/convert_yam_subset.py`针对原ABC来源；HIL的HDF5+三视频不能因为维度相同直接套原转换器。检索已有HIL导出工具并核对输入schema、动作时间配对、过滤缺口、H50和来源映射；缺少时补薄转换适配，不复制trainer。norm继承固定基线不等于重算混合norm，provenance校验需要如实表达两种来源。

Pi首轮工程候选：batch32、全量微调、peak5e-6、warmup100、末端1e-6、每1000步保存，H与RTC/资产均继承合同。尚需GPU验证；其他模型不继承该配方。manifest明确parent checkpoint、原始训练数据清单，不使用可变latest或只用“20h/90000”名称作身份。

数据合同映射：`observation_state`配对图像；审核并按时间对齐的`submitted_action`作absolute专家目标。`human_action`是未经相对补偿的leader输入；`submitted_action`不是物理到位；缺失`__valid`占位零不可训练。`expert_valid`、phase/source、软接管和同步共同决定片段，不强制policy_valid。省略等待导致的缺口必须切断，RL原始转移另保留。

## 参考研究

完整参考与适用边界仍归项目手册第8节。安装后不复制一份项目文献结论缓存。

- [High-DoF VLA Post-Training](https://arxiv.org/html/2609.19666v1)：每轮50纠正、新/历史1:1，从官方base重训；不同于本项目固定任务基线。
- [Hand-in-the-Loop](https://arxiv.org/html/2605.15157v2)：任务SFT后纠正适配，新/旧0.5:1；不是多轮固定M0优越性证明。
- [IWR](https://arxiv.org/html/2012.06733v1)：介入/自主样本等量，不等同原示范/纠正等量。
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html)：聚合访问状态的专家标注，不规定现代checkpoint初始化。
- LLM回放论文的5%/25%等数值不能直接移植给机器人。配比与预算对照采用项目手册，不宣称通用最优比例。
