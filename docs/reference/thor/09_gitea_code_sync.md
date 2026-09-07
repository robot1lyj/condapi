# Thor · 09 Gitea 代码同步

方向为工作站 `main` → Gitea `origin/main` → Thor `main`。GitHub 仍由工作站作为备份推送；Thor 只从 Gitea 拉取，不配置 GitHub 写入，也不进行后台双向覆盖。

## 首次授权状态（2026-09-07）

Thor 已生成 `/home/wuyan-lyj/.ssh/id_ed25519_condapi_gitea`，私钥仅保留在 Thor，权限 600。公钥由用户添加到 Gitea `wuyan_lyj/conda_pi` 仓库的只读部署密钥；无需给写入权限。

指纹为 `SHA256:J1jDdQ+DUQHweYsjA5xSYyCi8Rui2O8Op01FuqF9Pis`。工作站公钥副本 `/home/wuyan-lyj/thor-system/gitea/thor-condapi.pub`；不把私钥、密码或 token 提交 Git。

Thor `/home/wuyan-lyj/condapi/.git` 已初始化，仓库局部 `core.sshCommand` 使用这把专用密钥及独立 known-hosts 文件，Gitea 主机键已与工作站已信任的键核对。`origin` 为 `ssh://git@192.168.110.142:2222/wuyan_lyj/conda_pi.git`，`pull.ff=only`。首次 `ls-remote` 仍为公钥未授权，**尚未 fetch/checkout，没有有效 HEAD**，不能运行普通快进同步冒充首次接入。

用户添加公钥后，先验证 `git ls-remote origin refs/heads/main`；首次将 rsync 副本接入 Git 时，先保留并核对现有代码差异，再建立一致工作树，不用 `reset --hard` 或直接覆盖掩盖差异。模型、录像、结果都位于独立的 `thor/pi` 目录，不纳入代码同步。

## 首次接入完成后的日常同步

在工作站项目根目录运行：

```bash
git push origin main
git push github main
python3 scripts/thor/sync_code.py --host thor-usb
```

脚本要求两端均为干净 `main`，本地提交已发布到 Gitea，且 Thor 的 `origin` 与工作站一致；只 fetch 和 `merge --ff-only`，最后检查提交哈希。遇到未提交变更、分叉、远端在同步期间改变或首次 HEAD 缺失时退出，不 stash、不 reset、不覆盖。Wi-Fi 管理可改为已配置的 `--host thor`。

同步源码不等于更新运行中的模型：容器镜像不会自动重建/重启，模型权重不会复制，功耗模式不会改变。新镜像仍需按精度/延迟对照后再切换 Pi 系列服务版本。
