#!/usr/bin/env bash
# Run under a local persistent process. Only our new archive/installation are touched.
set -euo pipefail
archive=/home/wuyan-lyj/condapi-env-transfer/condapi-yam-20260907.tar.gz
expected=8005d7f2639cb31e3e67018cfbca08b93b33d76de99366497f600e3ecbe511ba
remote_archive=/home/wuyan/lyj/YAM/env-transfer/condapi-yam-20260907.tar.gz
remote_installer=/home/wuyan/lyj/YAM/env-transfer/install_packed_env.sh
prefix=/home/wuyan/.conda/envs/condapi-yam
code_root=/home/wuyan/lyj/YAM/YAM_code
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec 9>/home/wuyan-lyj/condapi-env-transfer/migration.lock
flock -n 9 || { echo 'Migration already running'; exit 1; }
scp "$script_dir/install_packed_env.sh" "yam-server:$remote_installer"
rsync -rt --partial "$(dirname "$archive")/repair-wheels/" yam-server:/home/wuyan/lyj/YAM/env-transfer/repair-wheels/
for attempt in 1 2 3 4 5; do
  echo "UPLOAD attempt=$attempt $(date -Is)"
  if rsync -t --partial --append-verify --info=progress2 --timeout=120 \
      -e 'ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3' \
      "$archive" "yam-server:$remote_archive"; then
    break
  fi
  [[ $attempt != 5 ]] || exit 1
  sleep 10
done
echo "UPLOAD_COMPLETE $(date -Is)"
# Once this detached remote job starts it no longer depends on the local host.
ssh yam-server "tmux new-session -d -s condapi-env-install 'bash $remote_installer $remote_archive $expected $prefix $code_root > /home/wuyan/lyj/YAM/env-transfer/install.log 2>&1'"
echo 'REMOTE_INSTALL_STARTED: /home/wuyan/lyj/YAM/env-transfer/install.log'
