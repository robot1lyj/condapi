# 用户材料与项目历史证据

本轮仅只读整理；没有重新分类、运行视觉模型、训练 RL、操作 Thor 或 3588。下面是历史审计的来源快照，不替代原 owner，也不表示当前服务状态。

## 已有研究决定

- 用户确认：14 集均为 20 万步 Pi checkpoint，自主运行，无 HIL，录到可作为任务终止的现场。
- 已选探索方向：冻结 Pi0.5、PLD 式残差 SAC＋价值初始化；整段抓取修正，左右臂覆盖，成功后交还 Pi 放置。
- 用户现阶段目标：跨任务可复用的 RL 框架；奖励模型可随 rollout 持续评分；不用快速人工接管作为必需条件。
- 2026-10-12 最新范围：本对话负责论文，先调研，想法和实验逐步确定，不强求现在找到创新。

## 可追溯审计

来源 worktree：`/home/wuyan-lyj/.codex/worktrees/parts-rollout/condapi`。

记录文件：

- `docs/reports/rl/parts-grasp-review14-20261010/README.md`，其中“当前资格修正与审核导入（2026-10-12）”一节。
- 同目录 `lift-qualification-summary.json`、`reclassification-v3-validation.json`。
- 原始数据：`/home/wuyan-lyj/condapi-data/rl/lego-rollout14-20261010-parts-v1`。
- 修订候选：`/home/wuyan-lyj/condapi-data/rl/lego-grasp-reclassified14-20261011-v3r3`。
- 资格侧车：`/home/wuyan-lyj/condapi-data/rl/lego-grasp-lift-audit14-20261012-v1`。

历史记录的源候选 SHA256：`7426b4973a0a4d3f367e70e398838bd1c9dd2a51e2e682322dbb4d2846948b0c`。

历史记录的用户审核导出 SHA256：`8249f3d28bb8c4030574c62cfebfabf837af999ead992a53c859cb0a74cbc582`。

本轮读到了 README 和资格摘要，未再次对全部数据逐成员验哈希；这些哈希是原审计的来源标识。

## 论文可使用与不可使用的数字

| 数字 | 含义 | 禁止解释 |
|---|---|---|
| 14 | 自主 rollout episode 数 | 14 个独立训练种子 |
| 342 | 闭爪候选提案 | 342 块独立积木 |
| 327 | 满足本批抬升/图像资格的候选 | 327 个独立完整任务 trial |
| 136 / 191 | 合格集合中的 held/empty 标签提案 | 已测 reward 模型的正确预测数 |
| 15 | 缺少合格核验证据而排除 | 已知空抓 15 次 |
| 13 | 用户已审核的主动选样片段 | 随机独立测试集 |
| 100% 分类完成 | 提案得到处置/分类 | 100% 检测准确率 |

E04-L015 的闭爪后核验区间没有抬升，下一次 L016 才抬升；这证明该条旧失败理由不成立，不证明所有失败原因都已定位。

原自动 Qwen 结果与助手最终判断的一致性是同批调试统计，不能当准确率；助手复核与用户抽检也不能证明完全无人参与。E01 曾参与感知调试。初期 checkpoint 资产来源冲突需在正式方法 provenance 中核实，用户 checkpoint 声明和文件来源证据分开保留。

## 关于零残差

Pi-only 意味着没有主动 residual 行为覆盖，而非可以随意重跑 Pi 并把差值当已执行 residual。若未记录原参考 chunk，不能用另一轮随机推理重建它。已有轨迹支持测量、行为附近的价值诊断；不能直接证明未尝试修正的效果。
