# 离线服务器 Docker 部署指南（openpi_dev_sudo）

本指南面向“离线服务器 + Docker 部署”的场景：在有网机器构建镜像与准备项目文件，迁移到离线服务器后直接启动即可使用。

## 1. 有网机器：拉取代码与构建镜像
```bash
git clone --recurse-submodules <repo_url> openpi
cd openpi

# 构建自包含开发镜像（含依赖 + 代码）
docker build -f scripts/docker/dev.Dockerfile -t openpi_dev_sudo .

# 导出镜像文件，准备迁移
docker save openpi_dev_sudo -o openpi_dev_sudo.tar
```

如需离线使用模型权重，建议提前下载到默认缓存目录 `/share/home/linyongjia/.cache/openpi/openpi-assets`，之后整体拷到离线服务器并自动映射为容器内的 `/root/.cache/openpi/openpi-assets`。

## 2. 有网机器：打包项目文件
```bash
cd ..
tar -czf openpi_repo.tar.gz openpi
```

## 3. 迁移到离线服务器
将 `openpi_dev_sudo.tar` 与 `openpi_repo.tar.gz` 拷贝到离线服务器（U盘/内网均可）。

## 4. 离线服务器：导入与启动
```bash
# 导入镜像
docker load -i openpi_dev_sudo.tar

# 解压项目
tar -xzf openpi_repo.tar.gz
cd openpi

# 启动容器（默认挂载当前代码目录 /app，默认端口 6666）
scripts/docker/run_dev.sh
```

默认会把仓库挂到 `/app`，因此你可以在宿主机修改代码并立即生效。  
脚本默认以 linyongjia 用户运行（UID:GID=1110:1011）。  
非 root 运行时会自动把容器缓存路径切换到 `/openpi_cache`，避免 `/root` 权限问题。  
若需要后台运行：`scripts/docker/run_dev.sh -d`。
如果看到提示 `I have no name!`，说明镜像里没有该用户条目，请重新构建镜像后再启动。

默认已映射以下路径（无需额外参数）：
- 模型缓存：`/share/home/linyongjia/.cache/openpi/openpi-assets` -> 容器 `/openpi_cache/openpi-assets`
- LeRobot 数据：`/share/home/linyongjia/data` -> 容器 `/data`（并设置 `HF_LEROBOT_HOME=/data`）
- 训练输出：`/share/home/linyongjia/output/openpi` -> 容器 `/output/openpi`（并设置 `OPENPI_OUTPUT_DIR=/output/openpi`）
若使用 `--as-root`，模型缓存会改为容器 `/root/.cache/openpi/openpi-assets`（仍使用同一个宿主机目录）。

## 5. 一条命令完成启动（进入 bash + 用户 + 端口 + GPU + 挂载）
下面是一条“全量版”启动命令，直接进入容器 bash。你只需要按需替换路径和 GPU 数即可：
```bash
scripts/docker/run_dev.sh \
  --mount /share/home/linyongjia/lyj/openpi \
  --gpus all \
  -p 6666 \
  -- /bin/bash
```

参数说明（都在这一条里）：
- `--mount`：挂载你的项目目录到容器 `/app`。
- `--gpus`：默认用全部 GPU（`--gpus all`），也可用 `--gpus 0,1` 指定设备。
- `-p 6666`：映射端口；默认也是 6666，这里显式写清楚。
- `--bind`：额外挂载文件或目录（可重复）。`/data` 已默认映射，无需再写。
- `-- /bin/bash`：进入容器交互式 bash。

如需显式指定用户，可在同一条命令里加：
```bash
--user 1110:1011
```
默认已是该用户，通常无需额外参数。如 UID/GID 不一致，请手动改成实际值。

如果你不想写完整命令，最低限度也可以这样（默认端口 6666、默认进入 bash）：
```bash
cd /share/home/linyongjia/lyj/openpi
scripts/docker/run_dev.sh --gpus all
```

如果之前权限已经被 root 改了，可在宿主机修复：
```bash
sudo chown -R "$USER":"$USER" /share/home/linyongjia/lyj/openpi
```

如果需要在容器里一次性删除项目目录（避免宿主机权限问题），可用下面命令挂载父目录并以 root 删除。请确认路径无误：
```bash
OPENPI_DATA_HOME=/tmp/openpi-assets LEROBOT_DATA_DIR=/tmp/lerobot OPENPI_OUTPUT_DIR=/tmp/openpi-output \
NAME=openpi_dev_rm scripts/docker/run_dev.sh \
  --as-root --no-gpu --host-net --mount /share/home/linyongjia/lyj -- \
  /bin/bash -lc 'rm -rf /app/openpi'
```

## 6. 在容器中运行服务/实验
进入容器后可按需运行（注意端口与映射一致）：
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/openpi_cache/openpi-assets/checkpoints/pi05_libero \
  --port 6666
```

也可以在启动时直接执行命令：
```bash
scripts/docker/run_dev.sh -p 6666 --gpus all -- \
  uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/openpi_cache/openpi-assets/checkpoints/pi05_libero \
  --port 6666
```

## 7. 训练输出位置（checkpoint）
默认训练输出在容器 `/app/checkpoints/<config>/<exp_name>`。  
如果你希望输出到默认映射的 `/output/openpi`，请在训练命令中加：
```bash
uv run scripts/train.py pi05_libero \
  --exp-name my_experiment \
  --checkpoint-base-dir /output/openpi
```

## 8. 训练示例（JAX）
下面给出一个完整训练示例（假设已准备好数据与配置）：
```bash
# 1) 计算归一化统计（按配置名）
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 2) 启动训练（输出到 /output/openpi）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero \
  --exp-name my_experiment \
  --checkpoint-base-dir /output/openpi
```
训练产物位置：`/output/openpi/pi05_libero/my_experiment/`

## 9. 数据缓存目录（权重/下载）
默认映射 `/share/home/linyongjia/.cache/openpi/openpi-assets` 到容器 `/openpi_cache/openpi-assets`。  
如需自定义目录：
```bash
OPENPI_DATA_HOME=/data/openpi_assets scripts/docker/run_dev.sh --gpus all
```

如需自定义 LeRobot 数据目录：
```bash
scripts/docker/run_dev.sh --lerobot-data /data/lerobot
```

## 10. 需要新增挂载时的处理（方案 A：保存环境并重建）
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
  --gpus all
```

说明：
- `docker commit` 只会保存容器内文件系统，不会包含挂载的目录。
- 镜像标签建议加日期，方便追溯与回滚。
