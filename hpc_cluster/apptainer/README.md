# apptainer -- Apptainer/Singularity container runtime, contrasted with Docker

**Status: code-complete, toolchain-gated, UNRUN. Empirically confirmed
this session (not assumed) that Apptainer, unlike Slurm, DOES have a
real Homebrew formula -- but real container execution stays Linux-gated
regardless.**

## What this measures

PLAN.md Phase 22 step 9: a real Apptainer definition file wrapping one of
this repo's binaries, contrasted directly against Phase 16's Docker
approach (`Dockerfile` at repo root) on no-daemon/no-root/direct
bind-mount GPU/MPI passthrough grounds -- Kurtzer, Sochat & Bauer
(2017)'s original design rationale for why Singularity/Apptainer exists
as a SEPARATE tool from Docker for HPC specifically, not a Docker clone.

`serving_daemon.def` wraps the exact same binary
`containers/Dockerfile` (Phase 16 step 1) already wraps --
`inference_serving/serving_backend/serving_daemon`, built via the
identical `cmake --preset debug --target serving_daemon` invocation --
deliberately kept identical so the two container definitions are a fair
contrast of the CONTAINER TECHNOLOGY, not two differently-scoped builds.

## Toolchain gate (empirically verified, not assumed)

Unlike Slurm (see `slurm_jobs/README.md`, which found NO real Homebrew
formula exists at all), `brew info apptainer` confirms a real, current,
bottled formula DOES exist on macOS: `apptainer` 1.5.3 ("Application
container and unprivileged sandbox platform for Linux", old name
`singularity`, real dependencies `libseccomp` + `squashfs`, no build
errors reported in the last 30 days). **Not installed this session** --
no explicit ask/grant for it, same standing no-new-local-installs policy
covering every toolchain in this phase.

**Even if installed, this would not close the real gap.** Apptainer's
own formula description says "for Linux": its core sandboxing mechanism
is UNPRIVILEGED LINUX USER NAMESPACES, a kernel feature with no macOS
equivalent -- structurally the same class of gap Docker Desktop papers
over by running a hidden Linux VM under the hood. Installing `apptainer`
via brew here would give real CLI tooling (enough to validate
`serving_daemon.def`'s SYNTAX) but not real container EXECUTION -- that
stays genuinely Linux-gated the same way Docker itself is, unlike
`open-mpi`/`libomp` (steps 1-2), which run their real payload natively
on macOS once installed.

## The real contrast, per Kurtzer/Sochat/Bauer's design rationale

| | Docker (`Dockerfile`, Phase 16 step 1) | Apptainer (`serving_daemon.def`) |
|---|---|---|
| Daemon | `dockerd` background daemon, root-owned | None -- runs as a direct child process of the invoking shell |
| Privilege model | Container processes run as root inside the container by default (rootless mode exists but isn't Docker's default) | Unprivileged by design -- the whole point of Apptainer's user-namespace approach: a container process runs as the SAME uid that launched it, no root escalation anywhere in the path |
| Multi-tenant HPC fit | Requires either root on the host or a rootless daemon setup per-user -- awkward on a shared cluster where users don't have root | Designed FOR shared multi-tenant clusters from the start -- no daemon means no shared root-owned process multiple users' jobs would otherwise need to trust |
| GPU/MPI passthrough | `--gpus` flag + NVIDIA Container Toolkit (a separate daemon-integration layer); MPI needs careful `--network`/`--ipc` flag tuning | Direct bind-mount of host GPU device files/MPI libraries into the container's namespace -- no separate toolkit layer, designed specifically for Slurm's `srun`-launched, per-job container model |
| Image format | Layered, daemon-managed image store | Single-file `.sif` (Singularity Image Format) -- trivially copyable to compute nodes with no daemon to register it with, a real fit for Slurm's `sbatch`-launched jobs on ephemeral compute nodes |

## Results
TODO: run once Apptainer is installed AND real Linux container-execution
hardware exists (the second condition is the one that actually matters --
see "Toolchain gate" above).

| Check | Expected |
|---|---|
| `apptainer build serving_daemon.sif serving_daemon.def` succeeds | TODO |
| `apptainer run serving_daemon.sif` serves real TCP connections identically to `docker run` against the same `cmake` target | TODO |
| No root/daemon required to run `apptainer run` (contrast with `docker run` needing `dockerd`) | TODO |
| GPU device bind-mount works without a separate NVIDIA Container Toolkit layer, once Phase 3's real CUDA kernels exist | TODO |

## Hardware/toolchain notes
- Required: real Linux (Apptainer's core sandboxing needs Linux user
  namespaces -- no macOS path, confirmed via the formula's own
  description, not assumed).
- `brew install apptainer` on this Mac would install real CLI tooling
  usable for `.def` file syntax validation but NOT real container
  execution -- see "Toolchain gate" above for why that's a genuine
  platform wall, not a missing-flag issue.
