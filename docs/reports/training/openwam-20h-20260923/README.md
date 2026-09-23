# OpenWAM YAM 20h 数据准备与训练参数待确认（2026-09-23）

## 已完成的数据准备

- 候选数据复用 Pi 20h 的固定选集：train 中 941 个完整 episode、2,160,813 帧、30 fps、20.007528 h；所有任务 prompt 为 `sort the legos into containers by color`。未混入独立 val。原选集为 `../rtc-base-20h-20260918/selection.json`，SHA256 `358ec6b0f21816b871e187559626b5a7043e2e16ba7bd2c9febecd1cdc34ee90`。
- 本目录 `episodes.json` 为上述选集的 941 个 LeRobot episode ID，SHA256 `317a9c16037ee547a98e8ec05d68d6ff7a1abe118716a29ee0baec7c28952925`。服务器副本在 `/home/wuyan/lyj/openwam-data/lego-20h-20260923/episodes.json`。
- 源数据为 `/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train`，LeRobot v3.0、14D、三路 RGB、30 fps。`meta/info.json` SHA256 `ba88887ff84544eb130708cdf8c0ce965b53f099fdad07286a57f41883211851`，与选集记录一致。源目录未修改。
- 使用服务器独立 `vla-openwam` 环境和 `adapters/openwam/prepare.py` 对所选 train episode 生成 `yam-train-stats.json`；本目录及服务器各保留一份，SHA256 `8a57de2f5546ef9c24547a96ee131a6bfe8d3d549aec8eced230b81bc0044867`。统计包含 941 个源 Parquet 文件及 metadata 的 SHA256，分别计算 action `joint` 和 state `joint_state` 的 min/max/mean/std；未读取 val 统计。
- 真实服务器 reader 对首、中、末窗口完成三相机解码，均为 9 帧 320×384 RGB 拼图，动作 `(32,14)`、状态 `(1,14)` 且有限。最后窗口只有 2 个有效动作时间点，mask 余下位置，未跨 episode。可用起点总数 2,159,872；按有效全局 batch 32，一遍约 67,496 次优化器更新。

## 配方决策前的阻塞项

- 官方 OpenWAM-Alpha foundation 是 80D 统一动作空间。用户决定保留官方完整 80D 动作/状态头：YAM 14D 按固定槽位 scatter，其他槽位填 0 并在动作损失中屏蔽；推理按同一映射 gather 为 14D。固定映射为 `[0,1,2,3,4,5,9,34,35,36,37,38,39,43]`，使左右臂落在原双臂区域、两夹爪沿用原 gripper 槽 9/43；其余 12 关节值占据原 EEF 槽，需要视为重新约定 YAM 语义，不能沿用 EEF 位置/旋转单位解释。无须派生 14D 权重，既有 80D foundation checkpoint 原件保持不变。服务器 Conda 环境已验证此映射的 14D→80D→14D 精确往返；真实首个 YAM 样本经完整视频读取后，动作 `(32,80)`、状态 `(1,80)`、动作有效 mask `(32,80)`，其余 66 槽全 0 且 loss mask 全假。用真实 train action/state 分别的 min-max 统计测试原生 deploy normalizer：状态输入与训练变换逐值差 0，模型归一化动作经 80D gather 后还原 raw14 最大差 `1.19e-7`；完整模型推理仍待 GPU 验证。
- 官方 README 推荐 8×80 GB GPU。基础权重文件 24,813,767,464 bytes，约 23.11 GiB；本服务器 4090 每卡 24,564 MiB。原生 ZeRO-2 复制权重到每卡，几乎无空间容纳激活/优化器，不能把“有 4 张卡”视为可训练。全量微调候选需要 ZeRO-3 参数分片、CPU optimizer offload、CPU 初始化、BF16、梯度检查点和每卡 microbatch 1；该组合尚未在本项目做 GPU 前反向、首个 Adam 状态分配及保存恢复验收。这里的“全量”指训练视频 DiT、ActionDiT、14D 头；上游默认冻结文本编码器、VAE 等预训练组件。
- Pi0.5 training-time RTC 使用动作位置级扩散时间，已执行前缀保持无噪且只对后缀计损失。OpenWAM 的 ActionDiT 当前只接受每样本一个时间条件，不能直接复用 Pi 的时间条件实现。本分支已在 OpenWAM 原生 loss 中加入动作前缀无噪条件和后缀损失掩码，并在采样器每次动作更新后重钳已提交前缀；`rtc_mode=off` 保留旧路径。用户已将延迟范围改为 **1–20 步**，对应 30 fps 下约 33–667 ms；尾端短 episode 会把延迟截到 `有效动作数-1`，至少保留一个学习目标。训练仍使用整条动作共用的时间调制，须通过容量/数值及前缀条件验收，不能声称与 Pi 数学实现相同。现有推理 DiT 缓存不是 RTC；LoRA 训练代码仍未接入。
- 2026-09-23 13:23 CST：用户 Pi 50h 作业 2167 占满 gpu001 四卡；gpu002 四卡由其他用户作业 2161 占用。不得附着现有作业做 OpenWAM GPU 测试。GPU 容量与吞吐仍未知。
- 源数据中的 `action` 按既有导出合同是绝对关节目标，夹爪 0=闭、1=开；raw-to-LeRobot 的单位逐值对照与实际机器人标定尚未完成。训练数据保持原始 14D 数值，部署前必须完成方向、单位、动作限位和反馈时序审核。

## 待用户确认的候选参数

已生成 `configs/native/openwam-yam-20h-rtc-capacity.json` 与本目录 `capacity.sbatch`，仅用于**8 microstep / 1 个优化器更新**的首轮跑通测试，不保存模型；4×4090、BF16、ZeRO-3、CPU optimizer offload、CPU 初始化、梯度检查点；每卡 microbatch 1、累积 8、有效全局 batch 32。官方 80D OpenWAM-Alpha foundation 权重 + 上述固定 14 槽映射；训练视频 DiT、ActionDiT 与 80D 动作/状态投影，冻结文本编码器、VAE；33 个原始时间点、视频 stride 4、32 步 action horizon、384×320 三相机拼图；训练集 min-max norm、seed 42；RTC 延迟 1–20 步。成功后再用独立输出目录做 100–200 microstep 稳态、完整权重保存/恢复、峰值显存和吞吐验收。若全量 OOM，改为单独的 LoRA + 80D 动作/状态投影可训练配方；该 LoRA 实现和保存/恢复仍需开发与测试，不应把架构内部 AdaLN-LoRA 当作现成 PEFT。

初始候选学习率为全量 `1e-5`，LoRA `5e-5`，其余优化器继承原生 AdamW `[0.9,0.95]` / weight decay `0.01`；正式训练总更新预算、保存周期尚未确定。注意 OpenWAM 的 `max_steps` / `save_steps` 计数是每进程 microstep，累积 8 时不能误报成优化器更新数。用户已授权先做服务器短跑通测试；正式长训练尚未启动。现有 `configs/native/openwam-yam.example.json` 只是接口示例。
