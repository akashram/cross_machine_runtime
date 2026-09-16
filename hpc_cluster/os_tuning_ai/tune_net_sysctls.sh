#!/usr/bin/env bash
# tune_net_sysctls.sh -- PLAN.md Phase 22 step 10: net.core.rmem_max/wmem_max
# for collective/RDMA-heavy throughput.
#
# Distinct from networking/'s own low-level RDMA/EFA tuning (which
# targets small, LATENCY-sensitive collective messages -- see
# networking/DESIGN.md) -- this targets the socket BUFFER SIZE ceiling
# for BULK, throughput-oriented traffic: a large ring/tree all-reduce
# transfer (networking/ring_allreduce, networking/tree_allreduce) or a
# bulk checkpoint write over NFS/RDMA (hpc_storage's own tuning, see
# READING_LIST.md's Phase 21 section) benefits from a large enough socket
# buffer that the kernel doesn't need to apply backpressure mid-transfer
# waiting for userspace to drain it -- the DEFAULT rmem_max/wmem_max
# (~212KB on stock Ubuntu) is sized for typical web-service request/
# response traffic, not multi-megabyte collective payloads.
# 16MB is a conservative, commonly-cited starting point for 10/25/100GbE
# bulk-transfer tuning (well above the default, well below the point of
# diminishing/negative returns from buffer bloat on a LAN-local,
# low-RTT collective network).
set -euo pipefail

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: net.core.rmem_max/wmem_max are Linux-only sysctls."
    echo "macOS has analogous but differently-named/scoped kern.ipc.maxsockbuf --"
    echo "not equivalent enough to safely automate here; see README.md."
    exit 1
fi
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

RMEM_BEFORE=$(sysctl -n net.core.rmem_max)
WMEM_BEFORE=$(sysctl -n net.core.wmem_max)
echo "Current net.core.rmem_max = ${RMEM_BEFORE}, net.core.wmem_max = ${WMEM_BEFORE}"

TARGET=$((16 * 1024 * 1024))  # 16MB
sysctl -w net.core.rmem_max="$TARGET"
sysctl -w net.core.wmem_max="$TARGET"
echo "Set net.core.rmem_max = net.core.wmem_max = ${TARGET} (16MB)"
echo ""
echo "To revert:"
echo "  sudo sysctl -w net.core.rmem_max=${RMEM_BEFORE}"
echo "  sudo sysctl -w net.core.wmem_max=${WMEM_BEFORE}"
