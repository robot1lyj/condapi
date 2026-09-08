# Evo-1 环境安装与 CPU 检查

日期：2026-09-08。安装规则和复现入口归 [环境手册](../../../02_installation_and_environment.md#evo-1--lerobot-独立环境)。此目录保存实际证据，不代表模型训练/部署验收。

| 位置 | 当前结果 |
|---|---|
| 本地工作站 | 安装完成；pip check、原生训练 --help、CPU 模块/处理器检查通过 |
| 服务器 | 安装完成；pip check、CPU模块/处理器检查通过，未初始化CUDA |
| GPU / 真实模型 | 未执行；没有下载模型权重或启动训练 |

本地报告：[workstation.json](workstation.json)，服务器报告：[server.json](server.json)。固定源码和wheel身份：[source.json](source.json)。完整106个已安装包（104个锁定wheel + bootstrap的pip/wheel）记录在审计JSON中；核心组合是 Python3.12.14、LeRobot0.6.2、Torch2.10.0/CUDA12.8、TorchVision0.25.0、TorchCodec0.10.0、Transformers5.5.4、Accelerate1.14.0。

服务器首次检查发现TorchCodec找不到Conda内FFmpeg动态库；通过新环境专属 `LD_LIBRARY_PATH=/home/wuyan/.conda/envs/vla-evo1-train/lib` 修正并通过检查，已用 `conda env config vars set` 持久化，激活环境或 `conda run` 生效。直接调用prefix内Python时也要显式传此变量。没有修改系统库或旧环境。2026-09-08 07:55 UTC附近复核保护作业2064仍RUNNING；这只是当时状态。

CPU脚本检查原生训练入口、Evo1Policy 模块和视频库可导入，使用三路16×16合成RGB、14D状态、合成min/max统计检查原生processor。输出为 `(1,50,14)`，左右夹爪均保持连续值，Torch CUDA未初始化。该结果不衡量真实图像预处理、YAM单位、checkpoint精度、GPU显存或训练速度。

安装过程没有更改服务器既有 `condapi-yam` 环境或运行中的代码检出。只对新prefix操作；服务器基础环境来自本地conda-pack，原先未完成的包缓存未修复/清空。pip网络下载慢时，改用限定两并发的curl并逐wheel校验SHA256，再执行离线pip安装。服务器 `/tmp` 中的wheel缓存不是持久资产，可由Git中的锁清单恢复；固定源码wheel留在家目录。

项目清理删除非YAM上游示例和空子模块；保留Pi模型实现、Thor测试报告、历史实验产物。JAX转换器迁到 `adapters/openpi/convert_jax_model_to_pytorch.py`，转换逻辑没有改变。代码与已有Thor/记忆系统回归当前168项通过（包括新下载完整性测试），有1项既有pynvml弃用警告。
