# MolmoAct2环境与接入检查

日期：2026-09-08。模型特有事项见 [接入说明](../../../reference/molmoact2_integration.md)，安装事实owner为 [02](../../../02_installation_and_environment.md)。

| 位置 | 当前结果 |
|---|---|
| 本地工作站 | 安装完成；pip check和CPU导入/原生配置/合成归一化检查通过 |
| 服务器 | 106个wheel哈希验证完成；独立Conda环境安装中 |
| 模型/GPU | 未加载checkpoint、未运行实际模型或训练 |

报告：[workstation.json](workstation.json)，补充实际 `conda run` + 环境动态库设置复核：[workstation-conda.json](workstation-conda.json)。固定LeRobot源码、Torch/CUDA/Python基础组合与 [Evo安装](../evo1-20260908/source.json) 相同，使用同一SHA256的源码wheel；Molmo环境额外加入PEFT0.20.0、SciPy1.18.1，独立锁见 `environments/molmoact2-wheels.lock.json`。共106个wheel加bootstrap中的pip/wheel，安装后108个Python包。

CPU检查调用真实原生模块和处理器的归一化步骤，验证14D双夹爪索引6/13的跳过归一化mask，合成输出 `(1,30,14)`，并运行原生优化器的单标量FP32更新。CUDA未初始化；未加载文本/图像tokenizer或完整输入processor，不代表真实YAM数据、训练精度或吞吐测试。

代码层新增模型声明、两个环境profile、实验占位，复用现有LeRobot后端。包含Molmo路由与拒绝未验收运行的回归测试，当前组合170项通过。模型注册保留planned，等待数据统计、动作语义、checkpoint和空闲GPU真实batch检查。
