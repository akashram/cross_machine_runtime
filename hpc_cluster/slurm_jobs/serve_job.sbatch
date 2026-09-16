#!/bin/bash
#SBATCH --job-name=cmr-serving-daemon
#SBATCH --partition=serving
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=/var/log/cmr/serving_daemon_%j.log
#
# Wraps inference_serving/serving_backend/serving_daemon's real
# long-running TCP inference process (trains the same toy transformer
# transformer_test.cpp validates, serves it over real TCP -- see that
# file's own header comment). SERVING_PORT below is the exact env var
# serving_daemon.cpp already reads (env_int("SERVING_PORT", 8080)) --
# offset by job ID so multiple serving jobs on the same node partition
# don't collide on one port.
#
# `interactive` QOS (see README.md's sacctmgr commands) gives this job a
# higher priority weight than the `training` partition's default jobs,
# so a short smoke-test/demo serving job doesn't sit queued behind a
# long training_worker job it has nothing to do with -- the entire
# reason this phase defines two QOS tiers instead of one.

set -euo pipefail

SERVING_PORT=$((8080 + SLURM_JOB_ID % 1000))
export SERVING_PORT
echo "serving_daemon: job ${SLURM_JOB_ID} listening on port ${SERVING_PORT} (node $(hostname))"
exec /opt/hpc_cluster/bin/serving_daemon
