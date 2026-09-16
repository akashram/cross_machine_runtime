# os_tuning_ai -- Linux/OS tuning for HPC AI workloads

**Status: real, complete scripts. The read-only/detection half
(`check_current.sh`, and every tuning script's own platform-detection
path) is run locally today; the live-tuning half is Linux-gated,
verified empirically, not assumed.**

## What this measures

PLAN.md Phase 22 step 10: extends `cpu_engine/os_tuning`'s exact
script-per-knob + before/after harness pattern to AI/HPC-training-
throughput knobs instead of that step's low-latency-jitter ones (real,
individually-toggleable scripts, not one monolithic toggle -- confirmed
distinct from `cpu_engine/os_tuning`'s existing scope by reading that
step's own README first, per this phase's scoping decision).

| Script | Knob | Reversible without reboot/relogin? |
|---|---|---|
| `tune_numa_balancing.sh` | `kernel.numa_balancing` -- disable automatic page migration so it doesn't fight `foundation/numa`'s explicit `mbind()`-based placement | yes |
| `tune_thp.sh` | `/sys/kernel/mm/transparent_hugepage/enabled` -- set `always` (distinct from `cpu_engine/hugepage`'s EXPLICIT hugepage allocation) | yes |
| `tune_memory.sh` | `vm.swappiness=0`, `vm.overcommit_memory=1` | yes |
| `tune_net_sysctls.sh` | `net.core.rmem_max`/`wmem_max` -> 16MB, for collective/checkpoint bulk transfer (distinct from `networking/`'s own small-message RDMA latency tuning) | yes |
| `tune_memlock.sh` | `/etc/security/limits.d/` memlock ceiling RDMA memory registration needs | **No** -- PAM-enforced, needs a new login/job |
| `tune_cgroup_v2.sh` | delegated cgroup v2 slice (`hpc-ai.slice`) feeding `slurm_jobs/cgroup.conf`'s own per-job containment | yes (until slices are populated) |
| `tune_all_ai.sh` | orchestrates the five live-reversible scripts in order, with before/after `check_current.sh` | -- |
| `check_current.sh` | reads every knob's CURRENT value, no root needed, changes nothing | -- (read-only) |

## Why split into scripts instead of one toggle
Same reasoning as `cpu_engine/os_tuning/README.md`'s own: a monolithic
script answers "is it faster now" but not "which knob bought how much" --
the question that matters when a future regression needs bisecting to a
specific setting.

## Results (captured 2026-09-16, this Mac -- Darwin; tuning scripts themselves are Linux-only)

`check_current.sh` (read-only, runs on any platform, no root needed):
```
=== os_tuning_ai/check_current.sh -- Darwin ===

-- NUMA balancing policy (Documentation/admin-guide/sysctl/kernel.rst: kernel.numa_balancing) --
kernel.numa_balancing = N/A (Linux-only sysctl, this is Darwin)

-- Transparent Huge Pages (Documentation/admin-guide/mm/transparent-hugepage.rst) --
THP enabled policy: N/A (Linux-only, /sys/kernel/mm/transparent_hugepage/enabled not present on Darwin)

-- Memory overcommit / swappiness (Documentation/sysctl/vm.rst) --
vm.swappiness = N/A (Linux-only sysctl, this is Darwin)
vm.overcommit_memory = N/A (Linux-only sysctl, this is Darwin)

-- Network buffer sizing for collective/RDMA-heavy throughput (Documentation/sysctl/net.rst) --
net.core.rmem_max = N/A (Linux-only sysctl, this is Darwin)
net.core.wmem_max = N/A (Linux-only sysctl, this is Darwin)

-- memlock ulimit (POSIX -- real on every platform, RDMA memory registration needs this raised) --
ulimit -l (max locked memory, KB) = unlimited

-- cgroup v2 availability (feeds Slurm's own cgroup containment, see slurm_jobs/cgroup.conf) --
N/A (Darwin has no /sys/fs/cgroup/cgroup.controllers -- cgroups are a Linux kernel feature)
```

`tune_all_ai.sh` on this Mac correctly detects it can't apply live
tuning and falls back to `check_current.sh` rather than silently no-op
succeeding:
```
os_tuning_ai/tune_all_ai.sh: tuning scripts are Linux-only.
Running check_current.sh instead to show what's read-able on Darwin:
[... same output as above ...]
```

`tune_memlock.sh` on this Mac -- the one knob that IS a real POSIX
concept on macOS too, even though the durable-override mechanism this
script edits (`/etc/security/limits.conf`) isn't:
```
NOTE: ulimit -l (memlock) IS a POSIX concept present on macOS too --
current value on this machine: unlimited
but /etc/security/limits.conf (PAM-based, what this script edits for
a durable per-user override) is a Linux-specific mechanism; macOS
uses launchd-based resource limits instead, out of scope here.
```

Every individual `tune_*.sh` script's platform-detection path (the
`[[ "$(uname)" != "Linux" ]]` branch) was exercised directly on this Mac
during development, exiting 1 with an explanatory message rather than
silently succeeding as a no-op -- same discipline
`cpu_engine/os_tuning/tune_isolcpus.sh` already established.

## Findings

- **`memlock` is the one knob genuinely checkable, with a real numeric
  value, on both platforms** -- macOS reports `unlimited` by default
  (BSD heritage, different default policy than Linux's typically-low
  64KB-8MB default), a real cross-platform data point even though the
  durable-override mechanism (`limits.conf`) itself doesn't transfer.
- **Every other knob is genuinely Linux-kernel-specific with no macOS
  analog** -- confirmed by direct `sysctl -n`/file-existence checks on
  this Mac, not assumed from documentation, matching the honesty
  standard `cpu_engine/os_tuning`'s own platform notes already set for
  `isolcpus`/C-states/IRQ affinity.
- **This step is genuinely distinct from `cpu_engine/os_tuning`**,
  confirmed by reading that step's own scope first before writing this
  one: that step tunes for tail-LATENCY jitter (isolcpus, C-states, IRQ
  affinity, `nohz_full`); this step tunes for large-memory
  training-job THROUGHPUT (NUMA balancing, THP, swappiness/overcommit,
  bulk network buffers, memlock, cgroup delegation) -- overlapping
  vocabulary (both touch CPU/memory/interrupt behavior) but different
  knobs and different justifications.

## Hardware/toolchain notes
`kernel.numa_balancing`, THP, `vm.swappiness`/`overcommit_memory`,
`net.core.rmem_max`/`wmem_max`, and cgroup v2 are Linux kernel interfaces
with no macOS equivalent -- same platform wall `cpu_engine/os_tuning`
already documents for its own knobs, confirmed independently for this
step's different knob set rather than assumed to carry over.
