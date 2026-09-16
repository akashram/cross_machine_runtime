#!/usr/bin/env bash
# tune_memlock.sh -- PLAN.md Phase 22 step 10: raise the memlock ulimit
# RDMA memory registration actually requires.
#
# WHY this matters specifically for RDMA (networking/rdma_v1's real EFA/
# libfabric work): registering a memory region for RDMA
# (ibv_reg_mr/fi_mr_reg) PINS those pages so the NIC can DMA into/out of
# them without kernel involvement per-transfer -- pinned pages cannot be
# swapped, so the kernel enforces RLIMIT_MEMLOCK as a ceiling on how much
# memory any one process can pin. The common Linux default (64KB, or
# sometimes 8-16MB depending on distro) is FAR below what a real
# multi-GB gradient/activation buffer needs to register for GPUDirect
# RDMA or EFA transfers -- an under-sized memlock ulimit doesn't degrade
# performance gracefully, it makes memory registration FAIL outright
# (`ibv_reg_mr: Cannot allocate memory`), a hard error a training job
# would otherwise hit only once it tries to actually use RDMA at scale.
#
# This is a per-user/per-process limit set via /etc/security/limits.conf
# (PAM-enforced at login), not a live sysctl -- requires a NEW login
# session (or `ulimit -l` set inside the launching shell/Slurm prolog) to
# take effect, unlike the sysctl-based scripts in this directory.
set -euo pipefail

if [[ "$(uname)" != "Linux" ]]; then
    echo "NOTE: ulimit -l (memlock) IS a POSIX concept present on macOS too --"
    echo "current value on this machine: $(ulimit -l)"
    echo "but /etc/security/limits.conf (PAM-based, what this script edits for"
    echo "a durable per-user override) is a Linux-specific mechanism; macOS"
    echo "uses launchd-based resource limits instead, out of scope here."
    exit 1
fi
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

LIMITS_FILE="/etc/security/limits.d/99-hpc-ai-memlock.conf"
TARGET_USER="${1:-*}"  # default: apply to all users; pass a username to scope it

echo "Current shell's memlock ulimit: $(ulimit -l)"

cat > "$LIMITS_FILE" <<EOF
# Installed by hpc_cluster/os_tuning_ai/tune_memlock.sh -- raises the
# memlock ulimit RDMA memory registration (ibv_reg_mr/fi_mr_reg) needs
# for real gradient/activation buffer sizes. See this script's own header
# comment for why the stock default is too low for this workload class.
${TARGET_USER} soft memlock unlimited
${TARGET_USER} hard memlock unlimited
EOF

echo "Wrote ${LIMITS_FILE}:"
cat "$LIMITS_FILE"
echo ""
echo "Takes effect on the NEXT login session (PAM-enforced), or set"
echo "'ulimit -l unlimited' directly in a Slurm job prolog/sbatch script"
echo "for immediate effect within that job."
echo ""
echo "To revert: sudo rm ${LIMITS_FILE}"
