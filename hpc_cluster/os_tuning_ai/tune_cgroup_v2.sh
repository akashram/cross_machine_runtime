#!/usr/bin/env bash
# tune_cgroup_v2.sh -- PLAN.md Phase 22 step 10: provision a delegated
# cgroup v2 slice for AI-workload resource isolation.
#
# Distinct from Slurm's OWN per-job cgroup containment
# (slurm_jobs/cgroup.conf's TaskPlugin=task/cgroup) -- that step
# constrains an INDIVIDUAL job's CPU/memory once Slurm launches it. This
# script provisions the HOST-level delegated slice Slurm's cgroup plugin
# needs to exist and be writable by the slurmd user BEFORE any job runs
# at all -- the two compose (this is the foundation, slurm_jobs/cgroup.conf
# is what Slurm builds on top of it), not duplicate functionality.
#
# Enables the `cpuset`, `memory`, and `pids` controllers on a dedicated
# delegated slice (`/sys/fs/cgroup/hpc-ai.slice`) -- `pids` specifically
# because a training job launching more helper/worker processes than
# expected (a runaway subprocess spawn, a leak in a data-loading worker
# pool) should be caught by the kernel's own pids.max ceiling rather than
# exhausting the host's process table, the same class of "degrade
# predictably" reasoning behind slurm_jobs/cgroup.conf's AllowedSwapSpace=0.
set -euo pipefail

CGROOT="/sys/fs/cgroup"
SLICE="${CGROOT}/hpc-ai.slice"

if [[ "$(uname)" != "Linux" ]] || [[ ! -f "${CGROOT}/cgroup.controllers" ]]; then
    echo "ERROR: cgroup v2 is a Linux-only kernel feature"
    echo "(${CGROOT}/cgroup.controllers not present on $(uname))."
    exit 1
fi
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo)"
    exit 1
fi

echo "Available root controllers: $(cat "${CGROOT}/cgroup.controllers")"

mkdir -p "$SLICE"
# Delegate the controllers this slice's children (Slurm's own per-job
# cgroups, created underneath this one) are allowed to use.
echo "+cpuset +memory +pids" > "${CGROOT}/cgroup.subtree_control" 2>/dev/null || true
echo "+cpuset +memory +pids" > "${SLICE}/cgroup.subtree_control"

echo "Provisioned delegated slice: ${SLICE}"
echo "Enabled controllers for children: $(cat "${SLICE}/cgroup.subtree_control")"
echo ""
echo "Point slurm.conf's cgroup plugin (CgroupMountpoint, if non-default) at"
echo "this slice, or configure slurmd to create its own job cgroups as"
echo "children of ${SLICE} rather than directly under ${CGROOT}."
echo ""
echo "To revert: sudo rmdir ${SLICE}  (only once empty of child cgroups)"
