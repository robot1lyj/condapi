# Xiaomi-Robotics-1 / XR-1 服务器环境与 checkpoint

观察日期：2026-09-24（Asia/Shanghai）。本记录区分用户本轮的安装请求与附件中描述的演示/训练步骤；附件只作为上游代码与合同参考，没有照着执行登录、下载演示集或开训。

## 检查点选择

选择官方通用后训练仓库 [`XiaomiRobotics/Xiaomi-Robotics-1-5B`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B)，revision `ee21d524b5c52ac961d941e1bc7d6d92836c3d5e`。官方模型说明将它作为 XR-1 后训练起点，因此适合先在自有数据上做微调；没有选择 RoboCasa、RoboCasa365 或 VLABench 等基准专用 checkpoint。

| 项目 | 结果 |
|---|---|
| 文件 | `model_states.pt` |
| 服务器路径 | `/home/wuyan/lyj/xiaomi-robotics-1/checkpoints/Xiaomi-Robotics-1-5B-ee21d524/model_states.pt` |
| 大小 | 10,226,684,862 bytes |
| SHA-256 | `94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7` |
| 下载/核验 | 从本地通过 SSH/rsync 传输；服务器摘要与 Hugging Face LFS SHA-256 相同 |

传输凭据保存在服务器 `checkpoints/Xiaomi-Robotics-1-5B-ee21d524/PROVENANCE.txt`。XR-1 源码快照放在 `source/xr1/`，来源提交为 `0dd7aef8dc87296246aae812a1f59ccb708e5546`；服务器 `source/upstream-source-manifest.sha256` 已核对快照文件。Qwen 依赖只缓存 `Qwen/Qwen3-VL-4B-Instruct` 的 config/tokenizer/processor 文件，revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，没有重复下载 Qwen 权重。Conda 激活后 `HF_HOME` 指向 `/home/wuyan/lyj/xiaomi-robotics-1/hf-home`。

## 环境

专用环境 prefix：`/home/wuyan/.conda/envs/xr1-posttrain`。安装脚本与包锁分别为服务器目录 `install/install_server_env.sh`、`install/xr1-posttrain-requirements.lock.txt`；安装日志 `install/logs/install-env.log`；完成标记 `INSTALL_COMPLETE`。

| 核心依赖 | 安装版本 |
|---|---|
| Python | 3.12.14 |
| PyTorch / torchvision / torchaudio | 2.8.0+cu128 / 0.23.0+cu128 / 2.8.0+cu128 |
| Torch CUDA build | 12.8 |
| Transformers | 4.57.1 |
| FlashAttention | 2.8.3（对应 Torch 2.8、CUDA 12、CPython 3.12、CXX11 ABI TRUE 的官方 wheel） |
| DeepSpeed | 0.18.9 |
| XR-1 package | 1.0.0 editable，来自上述固定源码快照 |

`pip check` 输出 `No broken requirements found.`。CPU 环境检查读取了 Torch 版本/CUDA build 与 ABI，不执行 CUDA device 初始化。`import decord` 通过。环境中未运行 XR-1 完整模型初始化、推理或训练；CUDA 驱动兼容性和 GPU 算子需在获准的 GPU 计算节点复核。安装期间 Slurm 作业 2176 `pi05-rtc-50h` 仍在运行，没有取消、暂停或干预。

## decord wheel 元数据修正

上游固定依赖要求 `decord==0.6.0`。服务器从镜像安装的官方 wheel 文件名为 `decord-0.6.0-py3-none-manylinux2010_x86_64.whl`，但 wheel 内 `decord-0.6.0.dist-info/WHEEL` 错写 `Tag: cp36-cp36m-manylinux2010_x86_64`，导致当前 pip 将其报告为“不支持此平台”。这是上游已记录的 [wheel 内外标签不一致问题](https://github.com/dmlc/decord/issues/356)，不是缺少 XR-1 依赖。

保留原始 WHEEL 元数据为 `install/decord-0.6.0.WHEEL.original`，把内部 tag 改为与官方发布文件名相同的 `py3-none-manylinux2010_x86_64`，同步重算 RECORD 后 `pip check` 通过。修复说明与幂等脚本在 `install/decord-wheel-tag-fix.txt`、`install/repair_decord_wheel_tag.py`；安装入口已在 `pip check` 前调用该脚本。

## YAM 适配边界

官方源码 `source/xr1/mibot/models/VLA/XR1.py` 将 `state_shape` 固定为 `(1, 60)`、`action_shape` 固定为 `(30, 60)`；XR-1 数据默认规范化数组也是 60D。当前 YAM 数据合同为 14D `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`。这两套张量不能直接互换；夹爪、关节与末端执行器状态如何投影、动作 60D 每一维的语义以及归一化统计都尚未确认。

因此当前状态是“官方 XR-1 环境与通用 checkpoint 已准备”，不是“XR-1 已支持 YAM”。下一步需要先基于 XR-1 数据格式和 YAM 实测 metadata 明确 60D 投影合同并审查数据/统计，再形成单独训练方案；本轮没有生成训练配置或命令、登录 W&B、下载 XR-1 演示数据或启动训练。GPU 资源上的模型初始化与数据 forward 仍待验收。
