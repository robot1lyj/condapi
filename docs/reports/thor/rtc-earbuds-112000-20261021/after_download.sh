#!/bin/bash
# Single authorized pipeline; no training and no operation on the control IPC.
set -euo pipefail
stage=/home/wuyan-lyj/thor/pi/probes/rtc-earbuds-112000-20261021-r1
exec 8>"$stage/pipeline.lock"
flock -n 8
echo WAIT_FOR_FROZEN_DOWNLOAD
exec 9>"$stage/download.lock"
flock 9
echo VERIFY_FROZEN_DOWNLOAD
/usr/bin/python3 "$stage/verify_download.py"
flock -u 9
echo CPU_PREPARE
/bin/bash "$stage/prepare.sh" > "$stage/prepare.log" 2>&1
echo EXCLUSIVE_GPU_AND_SAME_ADDRESS_DEPLOY
/bin/bash "$stage/deploy.sh" > "$stage/deploy.log" 2>&1
echo PIPELINE_COMPLETE_EARBUDS_112000
