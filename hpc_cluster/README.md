# Phase 22: HPC Cluster Systems Engineering

**Status: CODE COMPLETE (11/11 steps), 2026-09-16.** Cross-cutting,
same "real code, honestly toolchain-gated where it must be" convention
as every other hardware-gated phase in this repo. Unlike Phase 21 (no
simple hourly rental exists for VAST/WekaFS/Lustre-GPFS), and unlike
Phase 20 (real cloud QPU access exists but wasn't provisioned this
session), this phase's gates are almost entirely a LOCAL-INSTALL
decision (MPI/OpenMP/Apptainer) or a genuine macOS-vs-Linux kernel-
feature wall (Slurm, cgroups, most `/proc`/`/sys` tunables) -- see
`DESIGN.md` for the three-tier gate taxonomy this phase's own
`brew search`/`brew info` investigation surfaced.

## Overview

Scoped 2026-09-12 against a full HPC Software Engineer JD the user
pasted directly (MPI/OpenMP/GPU pipelines, Slurm-style scheduling,
CPU/GPU/NUMA/PCIe node-design awareness, rack-level power/cooling/
reliability engineering, HW/SW co-debug, containerized HPC runtimes).
Closes the gap: this repo has deep from-scratch understanding of
collectives, scheduling, and threading (it built ring/tree all-reduce
and multiple work-stealing/thread-pool schedulers by hand), but had
never called real MPI or OpenMP, never run a real cluster scheduler, and
had no rack-level power/cooling/reliability model at all -- the same
"hand-rolled first, framework-fluency second" pattern Phase 19 closed
for PyTorch/JAX, applied here to the classic HPC toolchain.

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| 1 | `mpi_allreduce` | Real MPI ring all-reduce vs. `MPI_Allreduce`, compared against `networking/ring_allreduce` | Code-complete, toolchain-gated (no `open-mpi` installed) |
| 2 | `openmp_port` | OpenMP reduction vs. `foundation::WorkStealingPool` | `WorkStealingPool` half run locally (3.0-3.2x on 4 threads); OpenMP half toolchain-gated (no `libomp`) |
| 3 | `slurm_jobs` | Real `slurm.conf`/`cgroup.conf`/`gres.conf` + `sbatch` scripts wrapping `training_worker`/`serving_daemon` | Code-complete, toolchain-gated -- confirmed empirically: real Slurm has NO Homebrew formula on macOS |
| 4 | `health_check` | Node health-check script, wired as Slurm's real `HealthCheckProgram` | Complete and run locally, both healthy and failure paths -- real bug caught and fixed |
| 5 | `rack_power_model` | Rack power/cooling capacity model, real typed interface for measured wattage | Complete and run locally |
| 6 | `reliability_model` | Cluster MTBF/MTTR/availability + redundancy tradeoff curve | Complete and run locally |
| 7 | `node_design` | NUMA/PCIe/topology node-design analysis, composing `foundation/numa`/`fpga_engine/pcie_latency`/`networking/topo_scheduler` | Written analysis, complete |
| 8 | `codebug_portfolio` | HW/SW co-debug case studies (raft SIGSEGV, cocotb DMA bug, PCA float32 bug) | Written portfolio, complete |
| 9 | `apptainer` | Apptainer definition file wrapping `serving_daemon`, contrasted with Phase 16's Docker | Code-complete, toolchain-gated -- confirmed empirically: real formula exists but needs Linux to execute containers |
| 10 | `os_tuning_ai` | Linux/OS AI-workload tuning scripts (NUMA balancing, THP, swappiness/overcommit, net buffers, memlock, cgroup v2) | Read-only/detection half run locally; live-tuning half Linux-gated |
| 11 | (reference material) | Rack & facilities physical integration | Filed in `READING_LIST.md`'s "Adjacent Engineering Disciplines" appendix, not code (same treatment as the EUV material) |

## Design highlights -- how the eleven steps connect

- **Steps 1-2 are direct comparisons against this repo's own existing
  hand-rolled primitives** (`networking/ring_allreduce`,
  `foundation::WorkStealingPool`), not fresh implementations measured
  in isolation.
- **Step 5's TDP fallback numbers are reused unchanged from
  `analog_engine/energy_model.h`**, and its real measured-wattage
  interface is designed to consume `gpu_engine/power`'s and
  `fpga_engine/xadc`'s real sensor calls automatically once either runs
  on real hardware.
- **Steps 3-4 are designed together, not sequentially bolted on**:
  `health_check/node_health_check.sh` is wired directly into
  `slurm_jobs/slurm.conf`'s real `HealthCheckProgram=` directive.
- **Step 9 wraps the IDENTICAL build target Phase 16's `Dockerfile`
  already wraps**, so its Docker-vs-Apptainer contrast is a fair
  technology comparison, not two differently-scoped builds.
- **Step 10 explicitly checked `cpu_engine/os_tuning`'s own scope
  first** before writing new scripts, confirming a genuinely distinct
  (throughput vs. latency-jitter) knob set rather than duplicating it.

## Real findings, not assumed conclusions

- **A three-tier toolchain-gate taxonomy, found empirically, not
  assumed** (see `DESIGN.md` section 1): `open-mpi`/`libomp` install and
  run natively on macOS (just declined this session); `apptainer`
  installs via Homebrew but can't execute real containers without
  Linux; real Slurm has no Homebrew formula on macOS AT ALL (the name
  collides with an unrelated network-monitoring tool). Three
  meaningfully different gates, confirmed via `brew search`/`brew info`
  for each, not inferred from "it's an HPC tool" as a blanket rule.
- **A real debug-vs-release measurement bug, caught and fixed** (step
  2): a `-O0` build's loop-shape mismatch produced a nonsensical 3.66x
  "speedup" for genuinely serial code; fixed and re-measured under
  `--preset release` to a believable, sub-linear 3.0-3.2x
  `WorkStealingPool` speedup on 4 threads. See `openmp_port/README.md`.
- **A real self-test environment-variable scoping bug, caught and
  fixed** (step 4): `node_health_check.sh --self-test`'s synthetic
  failure injection silently had no effect due to a one-shot env-var
  prefix not being re-read inside the already-evaluated function; caught
  by the self-test itself, fixed by making the threshold an explicit
  argument. See `health_check/README.md`.
- **The rack-scale reliability math produces a genuinely counter-
  intuitive, quantified result** (step 6): a 100-node cluster where each
  node individually fails only once every ~11 years still has SOME node
  failing roughly every 6 weeks at cluster scale -- the real,
  math-derived justification for treating node failure as routine
  infrastructure, not an exceptional path, in step 3's Slurm config and
  step 4's health-check tooling.
- **Both overcommit failure modes (power feed vs. cooling capacity) are
  independently triggerable**, not coupled (step 5) -- a rack can be
  within its power budget while exceeding its cooling system's capacity
  if PUE is high, and the model distinguishes the two rather than
  reporting one generic flag.
- **The rack power model is verified to actually consume a real
  measured-wattage input**, not just fall back to its literature TDP
  default, when one is available (step 5's explicit definition-of-done
  test case) -- and a reader that returns "no reading" (simulating a
  real sensor-call failure) correctly falls back to TDP rather than
  silently reporting zero watts.

See each step's own README for full methodology and captured output;
`hpc_cluster/DESIGN.md` for the phase-level design rationale.

## Hardware/toolchain notes

Steps 1-2 (MPI, OpenMP) need only a declined-this-session `brew install`
to close their gap -- no cloud hardware. Steps 3, 9, and the live-tuning
half of step 10 need real Linux (Slurm, real container execution, and
most `/proc`/`/sys` tunables are Linux kernel features with no macOS
equivalent, each confirmed empirically rather than assumed -- see each
step's own README). Step 5's real-wattage input has the same access gate
as `gpu_engine/power`/`fpga_engine/xadc` themselves (real GPU/FPGA
hardware, not introduced by this phase). See PLAN.md's hardware-
validation table, Phase 22 row.

## Next

Phase 22 was scoped alongside Phase 20 (Quantum Computing) and Phase 21
(HPC Storage Engineering) -- see `READING_LIST.md`'s Phase 20/21
sections for their own progress.
