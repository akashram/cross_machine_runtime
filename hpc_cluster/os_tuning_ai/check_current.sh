#!/usr/bin/env bash
# check_current.sh -- PLAN.md Phase 22 step 10: read the CURRENT value of
# every AI-workload tuning knob this step covers, without changing
# anything and without needing root. Runs on any platform today; on
# Linux it reads real values, on macOS it honestly reports which knobs
# have no equivalent -- same "document why, don't stub a no-op" discipline
# as cpu_engine/os_tuning/tune_isolcpus.sh's own platform-detection code.
set -uo pipefail

echo "=== os_tuning_ai/check_current.sh -- $(uname -s) ==="
echo ""

read_sysctl() {
    local key="$1"
    if [[ "$(uname)" == "Linux" ]] && command -v sysctl >/dev/null 2>&1; then
        sysctl -n "$key" 2>/dev/null || echo "N/A (not present on this kernel)"
    else
        echo "N/A (Linux-only sysctl, this is $(uname))"
    fi
}

read_proc_file() {
    local path="$1"
    if [[ -r "$path" ]]; then
        cat "$path"
    else
        echo "N/A (Linux-only, $path not present on $(uname))"
    fi
}

echo "-- NUMA balancing policy (Documentation/admin-guide/sysctl/kernel.rst: kernel.numa_balancing) --"
echo "kernel.numa_balancing = $(read_sysctl kernel.numa_balancing)"
echo ""

echo "-- Transparent Huge Pages (Documentation/admin-guide/mm/transparent-hugepage.rst) --"
echo "THP enabled policy: $(read_proc_file /sys/kernel/mm/transparent_hugepage/enabled)"
echo ""

echo "-- Memory overcommit / swappiness (Documentation/sysctl/vm.rst) --"
echo "vm.swappiness = $(read_sysctl vm.swappiness)"
echo "vm.overcommit_memory = $(read_sysctl vm.overcommit_memory)"
echo ""

echo "-- Network buffer sizing for collective/RDMA-heavy throughput (Documentation/sysctl/net.rst) --"
echo "net.core.rmem_max = $(read_sysctl net.core.rmem_max)"
echo "net.core.wmem_max = $(read_sysctl net.core.wmem_max)"
echo ""

echo "-- memlock ulimit (POSIX -- real on every platform, RDMA memory registration needs this raised) --"
echo "ulimit -l (max locked memory, KB) = $(ulimit -l)"
echo ""

echo "-- cgroup v2 availability (feeds Slurm's own cgroup containment, see slurm_jobs/cgroup.conf) --"
if [[ -d /sys/fs/cgroup ]] && [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
    echo "cgroup v2 mounted at /sys/fs/cgroup, controllers: $(cat /sys/fs/cgroup/cgroup.controllers)"
else
    echo "N/A ($(uname) has no /sys/fs/cgroup/cgroup.controllers -- cgroups are a Linux kernel feature)"
fi
