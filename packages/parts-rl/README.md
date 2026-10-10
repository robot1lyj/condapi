# PARTS 乐高抓取复现

本包按 [PARTS v2](https://arxiv.org/html/2609.21788v2) 公开方法独立实现。
[作者项目页](https://destiny000621.github.io/PARTS/) 在 2026-10-10 仍标注
Code Coming Soon，不称为官方源码或已完成实验复现。

当前完成的是夹爪动作合同、可回放的抓取监督状态机，以及 Thor SAM3
环境安装、正式权重转运与录像前向检查。尚无 TD3+BC learner、已训练 actor 或生产接入。
`packages/residual-rl` 保留此前 CalQL/SAC 实验，不是当前 PARTS 入口。

## 当前方法合同

- 冻结已经微调的 Pi0.5；接近、抬起、运输、放置仍由 Pi 提供动作。
- 左右夹爪独立专家，输出 H50×1 开度残差；每侧只有第 6 / 13 维可编辑。
  论文附录允许开度修正 ±0.50，最终开度投影到 `[0,1]`。
  这是论文候选合同，未通过现场动作验收，不能据此直接启用。
- 原始 RTC 前缀逐值保持；服务端 proposal 不当作执行回执。
  学习回放应使用客户端实际应用后的残差和实际执行 mask。
- 每次抓取独立 0/1 奖励；整轮正确分拣比例仅用于评估。
- 用户已确认抬升差阈值 **0.05 m**；结合腕部持物视觉连续保持 **1 s**。
  FK 参考点/向上轴、视觉区域及时间对齐仍须验证。
- 无法判断不是失败。遮挡、过期、epoch 中断进入 review，不伪造 0。
- `practice` 成功发出逻辑重置请求；`evaluate` 成功交还 Pi 继续分拣。
  重置实际执行只属于已有 YAM 客户端，本包不生成硬件控制代码。
- 后续 learner 应实现 chunk-level TD3+BC、成功残差 BC、可选失败零锚定、
  reference dropout、全成功加抽样失败的重训与重新部署。

## 环境与审核

Thor Dockerfile 在 `scripts/docker/thor/parts-vision.Dockerfile`，固定 NVIDIA
ARM64 PyTorch 基础镜像及 SAM3 revision。安装使用私有 venv，GPU 栈继承
NVIDIA 构建，NumPy <2 仅在该 venv 中覆盖。官方权重仓库的访问仍待审核；
用户提供公开镜像后，已核对 sam3.pt 与官方 metadata 的 SHA256 一致，
按固定 revision 下载并转运，真实前向状态以报告为准。
模型环境及本轮实际验收结果归 `docs/reports/thor/parts-vision-20261010/`。

工作站只做纯数组/协议检查；禁止本地训练及训练 smoke。
前向算子检查在 Thor 运行，随机初始化检查绝不生成真实奖励。
客户端实施要求先展示并经用户审核，不自动发任务。
