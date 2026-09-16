#!/usr/bin/env bash
# tune_numa_balancing.sh -- PLAN.md Phase 22 step 10.
#
# Sets kernel.numa_balancing=0 (disabled) for large-memory AI training
# jobs. WHY disable rather than tune it: Linux's automatic NUMA balancing
# periodically unmaps and re-faults pages to sample which NUMA node
# actually touches them, migrating hot pages toward the accessing node.
# That's a reasonable default for a general-purpose multi-tenant box, but
# actively FIGHTS a workload that already does explicit NUMA placement --
# exactly what foundation/numa/numa.h's NumaArena + bind_thread_to_node
# already do (mbind() a slab to a specific node, pin a thread to that
# node's CPU set). Automatic balancing's periodic page migration would
# undo that explicit placement mid-run, adding real page-fault overhead
# for zero benefit on a workload that already knows its own access
# pattern -- the opposite problem cpu_engine/os_tuning's isolcpus/
# nohz_full tuning solves for low-LATENCY workloads, applied here to
# large-MEMORY-footprint training jobs instead.
#
# Reversible without reboot (unlike cpu_engine/os_tuning/tune_isolcpus.sh's
# GRUB-based isolcpus).
set -euo pipefail

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: kernel.numa_balancing is a Linux-only sysctl."
    echo "macOS has no NUMA nodes at all (single-socket consumer hardware --"
    echo "see foundation/numa/numa.h's own documented platform note)."
    exit 1
fi
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

CURRENT=$(sysctl -n kernel.numa_balancing)
echo "Current kernel.numa_balancing = ${CURRENT}"
sysctl -w kernel.numa_balancing=0
echo "Set kernel.numa_balancing = 0 (disabled)"
echo ""
echo "To revert: sudo sysctl -w kernel.numa_balancing=${CURRENT}"
