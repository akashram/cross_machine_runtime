#!/bin/bash
#SBATCH --job-name=cmr-training-worker
#SBATCH --partition=training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/var/log/cmr/training_worker_%j.log
#
# Wraps distributed_training/training_worker's real process-per-rank
# binary (see that step's README -- already validated as 4 real,
# separate OS processes on this Mac, matching this job's --nodes=4
# exactly). RANK/WORLD_SIZE/PEER_HOSTS below are the SAME environment-
# variable convention training_worker.cpp already reads (see that file's
# own header comment) -- Slurm's SLURM_PROCID/SLURM_NTASKS map directly
# onto RANK/WORLD_SIZE, and `scontrol show hostnames` expands
# SLURM_JOB_NODELIST's compact range syntax into the real per-node
# hostname list training_worker's peer_hosts Channel constructor expects
# (one hostname per rank -- the same convention k8s/training/statefulset.yaml
# already builds from each pod's predictable per-ordinal DNS name).

set -euo pipefail

PEER_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
CKPT_DIR="/scratch/${USER}/cmr_ckpt_${SLURM_JOB_ID}"
mkdir -p "$CKPT_DIR"

srun --export=ALL,WORLD_SIZE="$SLURM_NTASKS",PEER_HOSTS="$PEER_HOSTS",BASE_PORT=46301,EPOCHS=100,CHECKPOINT_DIR="$CKPT_DIR" \
  bash -c 'RANK=$SLURM_PROCID /opt/hpc_cluster/bin/training_worker'
