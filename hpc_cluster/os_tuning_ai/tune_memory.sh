#!/usr/bin/env bash
# tune_memory.sh -- PLAN.md Phase 22 step 10: vm.swappiness + vm.overcommit_memory
# for large-memory training jobs.
#
# vm.swappiness=0: a training process's resident set (model params +
# optimizer state + activations) is exactly the working set it needs for
# the ENTIRE step it's currently computing -- swapping any of it out mid-
# step doesn't free memory for something more valuable, it just stalls
# the next access on disk I/O. Default swappiness (60) actively swaps
# warm-but-not-hot pages under memory pressure; 0 tells the kernel to
# avoid swapping application memory unless truly unavoidable (OOM is
# preferable to unpredictable multi-millisecond stalls in the middle of
# a training step -- same "degrade predictably rather than silently"
# reasoning slurm_jobs/cgroup.conf's AllowedSwapSpace=0 applies at the
# per-job cgroup level; this sysctl applies it host-wide).
#
# vm.overcommit_memory=1 (always overcommit): distributed training
# allocates large gradient/activation buffers whose PEAK combined size
# (all ranks' buffers, worst case simultaneously live) can look larger
# than physical RAM to a strict accounting heuristic (mode 0, the
# default) even when actual usage never gets there -- mode 0's heuristic
# can refuse a legitimate large mmap for a buffer that will, in practice,
# never be fully touched at once (e.g. many are zero-initialized and only
# sparsely written before use). Mode 1 removes that refusal, trusting the
# workload's own logic (gradient clipping, ZeRO-style sharding) to keep
# real usage bounded -- appropriate specifically for a KNOWN, TRUSTED
# training workload, not a general-purpose multi-tenant host where
# overcommit risk needs the heuristic's guardrail.
set -euo pipefail

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: vm.swappiness and vm.overcommit_memory are Linux-only sysctls."
    echo "macOS uses a dynamic pager with no user-tunable swappiness/overcommit knob."
    exit 1
fi
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

SWAPPINESS_BEFORE=$(sysctl -n vm.swappiness)
OVERCOMMIT_BEFORE=$(sysctl -n vm.overcommit_memory)
echo "Current vm.swappiness = ${SWAPPINESS_BEFORE}, vm.overcommit_memory = ${OVERCOMMIT_BEFORE}"

sysctl -w vm.swappiness=0
sysctl -w vm.overcommit_memory=1
echo "Set vm.swappiness = 0, vm.overcommit_memory = 1"
echo ""
echo "To revert:"
echo "  sudo sysctl -w vm.swappiness=${SWAPPINESS_BEFORE}"
echo "  sudo sysctl -w vm.overcommit_memory=${OVERCOMMIT_BEFORE}"
