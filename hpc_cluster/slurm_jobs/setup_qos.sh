#!/usr/bin/env bash
# setup_qos.sh -- real sacctmgr commands defining the two QOS tiers
# slurm.conf's `serving` partition (`QOS=interactive`) and PriorityWeightQOS
# reference. Run once against a live slurmdbd (needs SlurmDBD + a
# configured accounting database -- both Linux-only, same gate as the
# rest of this step).
#
# Priority=100 for `interactive` vs. Priority=10 for `training`: a 10x
# QOS-level priority gap. Combined with slurm.conf's own
# PriorityWeightQOS=10000 (the single largest per-factor weight in that
# file), this is the lever this cluster uses to let a short
# interactive/serving job reliably outrank jobs queued under the
# fairshare-heavy `training` QOS, not just tie-break within it.
set -euo pipefail

sacctmgr -i add qos training \
  MaxWall=48:00:00 \
  MaxTRESPerUser=cpu=32 \
  Priority=10

sacctmgr -i add qos interactive \
  MaxWall=04:00:00 \
  MaxTRESPerUser=cpu=8 \
  Priority=100

echo "QOS tiers configured: training (Priority=10, MaxWall=48h), interactive (Priority=100, MaxWall=4h)"
