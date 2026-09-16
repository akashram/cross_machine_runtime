#!/usr/bin/env bash
# tune_thp.sh -- PLAN.md Phase 22 step 10.
#
# Sets Transparent Huge Pages (THP) to "always" for large-memory AI
# training jobs. Distinct from cpu_engine/hugepage's EXPLICIT hugepage
# allocation (a program opts in via mmap flags/hugetlbfs) -- THP is the
# kernel TRANSPARENTLY backing anonymous memory with 2MB pages instead of
# 4KB, no application change needed. WHY "always" rather than the
# distro-common default "madvise": a training process's parameter/
# activation/gradient buffers are large, long-lived, densely-accessed
# allocations for the ENTIRE run -- exactly the profile that benefits from
# fewer TLB misses (a 2MB page covers 512x the address space of a 4KB
# page per TLB entry) with none of "always" mode's usual downside
# (background khugepaged compaction stalls hurting a bursty, short-lived-
# allocation workload) since a training process's allocations are neither
# bursty nor short-lived.
set -euo pipefail

THP_PATH="/sys/kernel/mm/transparent_hugepage/enabled"

if [[ "$(uname)" != "Linux" ]] || [[ ! -f "$THP_PATH" ]]; then
    echo "ERROR: Transparent Huge Pages is a Linux-only kernel feature"
    echo "($THP_PATH not present on $(uname))."
    exit 1
fi
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

CURRENT=$(cat "$THP_PATH")
echo "Current THP policy: $CURRENT"
echo always > "$THP_PATH"
echo "Set THP policy: always"
echo ""
echo "To revert to the common distro default:"
echo "  echo madvise | sudo tee $THP_PATH"
