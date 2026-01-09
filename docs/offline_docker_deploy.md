# 离线服务器 Docker 部署指南（openpi_dev）

本指南面向“离线服务器 + Docker 部署”的场景：在有网机器构建镜像与准备项目文件，迁移到离线服务器后直接启动即可使用。

## 1. 有网机器：拉取代码与构建镜像
```bash
git clone --recurse-submodules <repo_url> openpi
cd openpi

# 构建自包含开发镜像（含依赖 + 代码）
docker build -f scripts/docker/dev.Dockerfile -t openpi_dev .

# 导出镜像文件，准备迁移
docker save openpi_dev -o openpi_dev.tar
```

如需离线使用模型权重，建议提前下载到默认缓存目录 `/share/home/linyongjia/.cache/openpi/openpi-assets`，之后整体拷到离线服务器并自动映射为容器内的 `/root/.cache/openpi/openpi-assets`。

## 2. 有网机器：打包项目文件
```bash
cd ..
tar -czf openpi_repo.tar.gz openpi
```

## 3. 迁移到离线服务器
将 `openpi_dev.tar` 与 `openpi_repo.tar.gz` 拷贝到离线服务器（U盘/内网均可）。

## 4. 离线服务器：导入与启动
```bash
# 导入镜像
docker load -i openpi_dev.tar

# 解压项目
tar -xzf openpi_repo.tar.gz
cd openpi

# 启动容器（默认挂载当前代码目录 /app，默认端口 6666）
scripts/docker/run_dev.sh
```

默认会把仓库挂到 `/app`，因此你可以在宿主机修改代码并立即生效。  
脚本默认以 root 运行容器（匹配默认缓存路径在 `/root/.cache`）。  
若需要后台运行：`scripts/docker/run_dev.sh -d`。

默认已映射以下路径（无需额外参数）：
- 模型缓存：`/share/home/linyongjia/.cache/openpi/openpi-assets` -> 容器 `/root/.cache/openpi/openpi-assets`
- LeRobot 数据：`/share/home/linyongjia/data` -> 容器 `/data`（并设置 `HF_LEROBOT_HOME=/data`）

## 5. 一条命令完成启动（进入 bash + 用户 + 端口 + GPU + 挂载）
下面是一条“全量版”启动命令，直接进入容器 bash。你只需要按需替换路径和 GPU 数即可：
```bash
scripts/docker/run_dev.sh \
  --mount /share/home/linyongjia/lyj/openpi \
  --gpus 2 \
  -p 6666 \
  -- /bin/bash
```

参数说明（都在这一条里）：
- `--mount`：挂载你的项目目录到容器 `/app`。
- `--gpus`：GPU 数量（或用 `--gpus 0,1` 指定设备）。
- `-p 6666`：映射端口；默认也是 6666，这里显式写清楚。
- `--bind`：额外挂载文件或目录（可重复）。`/data` 已默认映射，无需再写。
- `-- /bin/bash`：进入容器交互式 bash。

如需以宿主机用户运行（避免 root 修改 `/app` 权限），可在同一条命令里加：
```bash
--user $(id -u):$(id -g)
```
注意：使用非 root 用户时，建议同时指定可写的容器缓存路径，例如：
```bash
OPENPI_DATA_HOME_IN_CONTAINER=/home/$(whoami)/.cache/openpi \
scripts/docker/run_dev.sh --user $(id -u):$(id -g)
```

如果你不想写完整命令，最低限度也可以这样（默认端口 6666、默认进入 bash）：
```bash
cd /share/home/linyongjia/lyj/openpi
scripts/docker/run_dev.sh --gpus 2
```

如果之前权限已经被 root 改了，可在宿主机修复：
```bash
sudo chown -R "$USER":"$USER" /share/home/linyongjia/lyj/openpi
```

## 6. 在容器中运行服务/实验
进入容器后可按需运行（注意端口与映射一致）：
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --port 6666
```

也可以在启动时直接执行命令：
```bash
scripts/docker/run_dev.sh -p 6666 --gpus 2 -- \
  uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --port 6666
```

## 7. 数据缓存目录（权重/下载）
默认映射 `/share/home/linyongjia/.cache/openpi/openpi-assets` 到容器 `/root/.cache/openpi/openpi-assets`。  
如需自定义目录：
```bash
OPENPI_DATA_HOME=/data/openpi_assets scripts/docker/run_dev.sh --gpus 1
```

如需自定义 LeRobot 数据目录：
```bash
scripts/docker/run_dev.sh --lerobot-data /data/lerobot
```

## 8. 需要新增挂载时的处理（方案 A：保存环境并重建）
Docker 不能给“已存在的容器”新增挂载，所以需要保存当前环境并重建容器。流程如下：
```bash
# 停止当前容器
docker stop openpi_dev

# 把当前容器保存成新镜像（保留已安装的依赖/修改）
docker commit openpi_dev openpi_dev_custom:2025-01-15

# 删除旧容器（释放名称）
docker rm openpi_dev

# 用新镜像重建，并加入新的挂载
IMAGE=openpi_dev_custom:2025-01-15 \
scripts/docker/run_dev.sh \
  --mount /share/home/linyongjia/lyj/openpi \
  --bind /new/data:/data \
  --gpus 2
```

说明：
- `docker commit` 只会保存容器内文件系统，不会包含挂载的目录。
- 镜像标签建议加日期，方便追溯与回滚。
