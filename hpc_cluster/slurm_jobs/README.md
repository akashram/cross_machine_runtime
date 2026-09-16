# slurm_jobs -- real Slurm cluster scheduling for this repo's real binaries

**Status: code-complete, toolchain-gated, UNRUN. Empirically confirmed
this session (not assumed) that real Slurm has NO Homebrew formula on
macOS at all -- see "Toolchain gate" below.**

## What this measures

PLAN.md Phase 22 step 3: real, complete `slurm.conf`/`cgroup.conf`/
`gres.conf` (GPU-aware) plus `sbatch` job scripts wrapping this repo's
actual long-running binaries (`distributed_training/training_worker`,
`inference_serving/serving_backend`'s `serving_daemon`) as real Slurm
jobs, with real backfill/fairshare/QoS/partition tuning parameters
actually set and justified -- not left at SchedMD's packaged defaults.

## Toolchain gate (empirically verified, not assumed)

`brew search slurm` on this Mac resolves to an UNRELATED tool: **"Yet
another network load monitor"**
(https://github.com/mattthias/slurm/wiki/), a small CLI bandwidth
visualizer that happens to share the name "slurm" -- NOT SchedMD's HPC
workload manager. Confirmed via `brew info slurm`: the formula's
description, homepage, and install-analytics numbers (~5-29 installs/
month, far below `open-mpi`'s ~3,700/month) all point at the network
monitor, not a workload manager. **The real Slurm has no Homebrew formula
on macOS at all.** This is a genuinely harder gate than every other
toolchain this phase checked -- `open-mpi` (step 1), `libomp` (step 2),
and `apptainer` (step 9) ALL have real, current, bottled Homebrew
formulas confirmed the same way (see those steps' own READMEs); Slurm
does not. Real Slurm needs SchedMD's `.deb`/`.rpm` packages or a source
build on real Linux cluster hardware (see PLAN.md's hardware-validation
table, Phase 22 row) -- this isn't a "just ask to install it" gap the way
steps 1/2/9/11 are, it's a genuine macOS platform wall.

## Design -- why these specific values, not defaults

- **`sched/backfill`** (not strict FIFO): this cluster's real jobs are
  deliberately heterogeneous in size -- a 4-rank
  `distributed_training/training_worker` job (long, wide) alongside a
  single-process `serving_daemon` smoke test (short, narrow). Strict
  FIFO would let a wide, long training job block a short serving job
  behind it even with idle capacity to run both -- exactly the gap
  backfill (Yoo, Jette & Grondona 2003) exists to close.
- **`PriorityWeightFairshare=100000`, an order of magnitude above
  `PriorityWeightAge`/`PriorityWeightJobSize`**: this repo's real
  workload mix is dominated by long-running jobs (`training_worker`
  real captured runs take multiple minutes even at toy scale -- see that
  step's README), where age-based priority alone would let one user's
  long job monopolize the cluster indefinitely.
- **Two QOS tiers** (`training`: `Priority=10`, `MaxWall=48h`;
  `interactive`: `Priority=100`, `MaxWall=4h`), set via `setup_qos.sh`'s
  real `sacctmgr` commands, matched by `slurm.conf`'s
  `PriorityWeightQOS=10000` (the single largest per-factor weight in the
  file) -- so a short `serving_daemon` smoke-test job reliably outranks
  jobs queued under the fairshare-heavy `training` QOS.
- **`TaskPlugin=task/cgroup` + `ConstrainRAMSpace=yes`/`AllowedSwapSpace=0`**:
  real per-job resource containment enforced by the kernel, not an
  advisory ulimit a runaway job could exceed unnoticed -- `cgroup.conf`
  cross-references step 10's `os_tuning_ai/tune_cgroup_v2.sh`, which
  provisions the delegated cgroup v2 slice Slurm's own per-job
  containment sits inside (the two steps compose, they don't duplicate).
- **`HealthCheckProgram=.../node_health_check.sh`, `HealthCheckInterval=300`**:
  wires in step 4's real health-check script directly (see
  `health_check/README.md`) -- a node that fails it is automatically
  drained by `slurmd`, standard `HealthCheckProgram` semantics.
- **`AutoDetect=nvml`** in `gres.conf`: real, documented SchedMD syntax --
  `slurmd` enumerates GPUs via NVML directly, the same library
  `gpu_engine/power/power_monitor.h` already wraps for this repo's own
  GPU power telemetry, reused here for device enumeration.

## Job scripts

- `train_job.sbatch`: `--nodes=4`, one task per node, wraps
  `training_worker`. `scontrol show hostnames "$SLURM_JOB_NODELIST"`
  expands Slurm's compact node-range syntax into the real per-node
  hostname list `training_worker`'s `peer_hosts` Channel constructor
  expects -- `SLURM_PROCID`/`SLURM_NTASKS` map directly onto
  `training_worker.cpp`'s existing `RANK`/`WORLD_SIZE` env-var reads (no
  new env-var convention invented -- reuses exactly what that step's
  README documents).
- `serve_job.sbatch`: single task, `--qos=interactive`, wraps
  `serving_daemon`, `SERVING_PORT` set to the exact env var
  `serving_daemon.cpp` already reads (`env_int("SERVING_PORT", 8080)`).

## Results
TODO: run once real Linux Slurm cluster hardware exists (see PLAN.md's
hardware-validation table). Once it does:

| Check | Expected |
|---|---|
| `sbatch train_job.sbatch` schedules 4 tasks across `cpu-node[01-08]` | TODO |
| `training_worker`'s loss trajectory over real Slurm-launched processes matches the existing `2.6317 -> 0.0008` baseline | TODO |
| `sbatch serve_job.sbatch` under `interactive` QOS starts ahead of a queued `training` QOS job of lower priority | TODO |
| A node made to fail `node_health_check.sh` is auto-DRAINed within one `HealthCheckInterval` (300s) | TODO |
| `cgroup.conf`'s memory constraint actually kills a job that exceeds `--mem` | TODO |

## Hardware/toolchain notes
- Required: real Linux cluster hardware, SchedMD Slurm built from source
  or installed via `.deb`/`.rpm` (no macOS path -- see "Toolchain gate"
  above), `munge` for `AuthType=auth/munge`, a configured `slurmdbd` +
  accounting database for `setup_qos.sh`'s `sacctmgr` commands.
- `gres.conf`'s `AutoDetect=nvml` additionally needs real NVIDIA GPU
  hardware + NVML -- same gate as `gpu_engine/` itself.
