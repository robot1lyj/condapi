# Evo-1 VLM 权重本地下载与服务器交接

观察日期：2026-09-16（Asia/Shanghai）

## 结论

已按当前 Evo-1 YAM 训练配置下载并交接 `OpenGVLab/InternVL3-1B-hf` 的完整基座 VLM，固定 revision 为 `014c0583a0d4bedf29fbe2dbff4f865eb998e171`。本地下载完成后逐文件校验，再传到服务器；服务器端 16 个文件全部与 ModelScope API 的 SHA256 一致。

服务器正式目录：

```text
/home/wuyan/lyj/evo1-install-2774d9b/models/OpenGVLab--InternVL3-1B-hf-014c0583
```

这是 Evo-1 的 VLM 起点，不是已经训练好的 YAM 策略：动作头仍需用 YAM 数据进行 Evo stage1→stage2 训练。没有下载 `zuoxingdong/evo1_libero`，因为它是 LIBERO 7D 语义 checkpoint，不能直接当作 YAM 14D 预训练。

## 来源与文件

当前 ModelScope 镜像 API 指向 revision `014c0583a0d4bedf29fbe2dbff4f865eb998e171`，对应 Hugging Face 仓库 [`OpenGVLab/InternVL3-1B-hf`](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)。官方 LeRobot Evo-1 文档指定该模型名，并说明首次运行会下载 VLM；同一文档的参考训练配方固定了该 revision。[LeRobot Evo-1 文档](https://huggingface.co/docs/lerobot/evo1)

下载入口使用 ModelScope 镜像：

```text
https://www.modelscope.cn/models/OpenGVLab/InternVL3-1B-hf/resolve/master/<file>
```

总文件数：16；总大小：`1,892,380,781` bytes。主要权重 `model.safetensors` 为 `1,876,478,048` bytes，SHA256 为：

```text
fdf8d51c7db5e31938300642b1bebce7f28592a2adefafc931cb8c21a8395517
```

完整的逐文件文件名、大小和 SHA256 保存在 [`model-transfer.json`](model-transfer.json)。本地副本位于：

```text
/home/wuyan-lyj/evo1-install-2774d9b/models/OpenGVLab--InternVL3-1B-hf-014c0583
```

## 传输与验证

直接单连接 rsync 速度出现明显抖动，因此采用了可续传的 64 MiB 分块传输：本地将 `model.safetensors` 分成 28 块，4 路 `rsync --append-verify` 传到服务器临时盘；服务器检查每块大小后按编号拼接，再对完整文件做 SHA256 校验，最后生成正式 `model.safetensors`。元数据文件随后单独同步并逐文件校验。

验证结果：

- 本地权重 SHA256 与 ModelScope API 一致。
- 服务器正式目录的 16 个文件逐一与 ModelScope API SHA256 一致。
- 服务器 Evo 环境在 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 下成功读取 `InternVLConfig`、`InternVLProcessor`、`Qwen2Tokenizer` 和 safetensors；权重含 733 个 tensor keys。
- GPU 节点 RTX 4090 上用服务器本地目录成功加载完整 VLM 权重：938,193,024 参数，BF16，位于 `cuda:0`；FlashAttention 请求为 `flash_attention_2`，检测可用。

GPU 加载原始日志：

```text
/home/wuyan/lyj/evo1-install-2774d9b/evo1-model-load-20260916.log
SHA256: 5203b5c0c9a94ebf274f1541c7dd4fb958cb8bc3aba775890da4afc215301050
```

日志有两个不影响加载的提示：`torch_dtype` 参数在当前 Transformers 版本建议改名为 `dtype`；checkpoint 同时保存了 tied embedding 和 lm_head 的不同值，因此未强制 tie。没有因此改动上游权重或 config。

服务器临时分块目录仍保留在 `/tmp/evo1-model-parts-20260916-a`；中断的单路上传断点也保留在：

```text
/home/wuyan/lyj/evo1-install-2774d9b/model-transfer-receipts-20260916/model.safetensors.rsync-partial-20260916
```

这些是交接证据，不是训练输入；正式训练命令应显式使用服务器正式目录，不能指向临时分块或断点文件。

## 尚未完成的验收

本次完成的是“基座 VLM 权重下载、传输、完整性和离线加载”验收；没有运行完整 Evo policy 的图像前向、真实 YAM batch、保存重载或训练循环。正式训练前仍需按项目 03/10 的门槛做真实三相机、14D 合同、stage1/2 和数值有限性检查。
