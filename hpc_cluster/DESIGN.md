# HPC Cluster Systems Engineering -- Design

## 1. Why this phase splits into three toolchain-gate tiers, not two

Every earlier hardware-gated phase in this repo (3/7/8/15/16) has one
gate: a piece of hardware or a toolchain either exists locally or it
doesn't. This phase found a THIRD tier by actually checking, not
assuming:

- **Genuinely installable on this Mac, just declined this session**:
  `open-mpi` (step 1) and `libomp` (step 2) both have real, current,
  bottled Homebrew formulas confirmed via `brew info` -- their only gate
  is the standing no-new-local-installs decision, not a platform limit.
- **Installable, but installing it wouldn't close the real gap**:
  `apptainer` (step 9) has a real, current, bottled Homebrew formula too
  -- but its own formula description says "for Linux," because its core
  mechanism (unprivileged Linux user namespaces) has no macOS
  equivalent. Installing it here would validate `.def` file syntax, not
  real container execution.
- **Not installable via Homebrew at all**: `slurm` (step 3) resolves on
  this Mac's Homebrew to an unrelated network-monitoring tool that
  happens to share the name -- the REAL SchedMD Slurm has no macOS
  formula. This is a harder gate than the other two tiers, confirmed
  the same empirical way (`brew search`/`brew info`), not assumed from
  Slurm being "an HPC tool" in the abstract.

Every one of these findings came from actually running `brew search`/
`brew info` this session rather than assuming "Linux-only tool = no
brew formula" as a blanket rule -- that blanket rule would have been
WRONG for `open-mpi`/`libomp`/`apptainer`, all three of which install
and (at least partially, for `apptainer`) run on macOS.

## 2. The reuse chain: six steps building on already-real components, not six new models

- Step 1 (`mpi_allreduce`) reimplements `networking/ring_allreduce`'s
  EXACT chunk-ownership convention (chunk `i` spans
  `[i*count/N, (i+1)*count/N)`, rank `r`'s phase-1 output is chunk
  `(r+1) % world_size`) with real MPI primitives instead of
  `netcommon::Channel` -- deliberately kept structurally identical so
  the two are comparable point-for-point once both run, not just "two
  things that both compute a sum."
- Step 2 (`openmp_port`) is compared directly against
  `foundation::WorkStealingPool` (Phase 1's real work-stealing pool),
  not a fresh baseline -- and, since the reduction kernel itself is the
  simplest possible parallel primitive, deliberately surfaces
  `WorkStealingPool`'s one real limitation this comparison makes
  concrete: no built-in `reduction()` clause, unlike OpenMP.
- Step 5 (`rack_power_model`)'s TDP fallback constants are the EXACT
  same numbers `analog_engine/energy_model.h`'s `digital_devices()`
  table already committed to (CPU 150W, GPU/A100 400W, NPU 2W) --
  reused, not re-guessed, so this model's literature default never
  quietly diverges from Phase 15/17's own numbers. Its real
  measured-wattage interface is designed to accept `gpu_engine/power`'s
  real NVML calls and `fpga_engine/xadc`'s real XRT calls the moment
  either runs on real hardware -- upgrading automatically rather than
  staying a permanent simulation, the exact shape PLAN.md asks for.
- Step 3 (`slurm_jobs`) wraps the TWO real long-running binaries this
  repo's earlier hardware-gap-closing work built: `training_worker`
  (Phase 16's real process-per-rank driver, already validated as 4 real
  OS processes on this Mac) and `serving_daemon` (Phase 16's real TCP
  inference process). No new binary was written for this step -- the
  gap it closes is real Slurm scheduling around binaries that already
  exist and already work.
- Step 4 (`health_check`) is wired directly into step 3's
  `HealthCheckProgram=` directive, not a standalone script -- the two
  steps are designed to compose from the start, not connected as an
  afterthought.
- Step 9 (`apptainer`) wraps the IDENTICAL `serving_daemon` build target
  `containers/Dockerfile` (Phase 16 step 1) already wraps, via the
  identical `cmake --preset debug --target serving_daemon` invocation --
  so the Docker-vs-Apptainer contrast in step 9's README is about the
  container TECHNOLOGY, not two differently-scoped builds.
- Step 10 (`os_tuning_ai`) explicitly checked `cpu_engine/os_tuning`'s
  own scope FIRST (reading that step's README) before writing new
  scripts, confirming this step's knobs (NUMA balancing, THP,
  swappiness/overcommit, bulk network buffers, memlock, cgroup v2) are
  a genuinely distinct throughput-focused set from that step's
  latency-jitter-focused set (isolcpus, C-states, IRQ affinity,
  `nohz_full`), not a redundant re-implementation.

## 3. Two real bugs, both caught by running the code, both worth keeping visible

**`openmp_port`'s debug-build measurement artifact.** The first captured
run (under the `debug` preset, `-O0`) reported a nonsensical 3.66x
"speedup" for the serial-fallback OpenMP path over a plain serial
baseline -- nonsensical because the OpenMP path, with no `libomp` linked,
compiles the `#pragma omp` down to a no-op and runs genuinely serially;
it has no mechanism to be 3.66x faster than an equally-serial baseline.
Root cause, traced rather than dismissed as noise: the serial baseline
used a range-based `for (float v : data)` loop while the OpenMP-pragma
function used an indexed `for (int i = 0; i < n; ++i)` loop -- at `-O0`,
with no inlining/vectorization to erase the difference, the two loop
SHAPES have measurably different per-iteration overhead, unrelated to
threading. Fixed by making both loops structurally identical and
re-measuring under `--preset release`: the serial-fallback path now
correctly reports ~1.0x, and `WorkStealingPool`'s real 4-thread speedup
lands at a believable, sub-linear ~3.0-3.2x. See
`openmp_port/README.md`'s full account. This is the same class of
lesson `analog_engine`/`ml/hyperband`'s own real bugs teach: a
suspicious number is a signal to root-cause structurally, not a result
to report as-is or explain away.

**`health_check`'s self-test environment-variable scoping bug.** The
first version of `node_health_check.sh --self-test` tried to inject a
guaranteed-to-fail disk-space threshold via
`HEALTH_CHECK_MIN_FREE_PCT=101 check_disk "/"` -- a one-shot env-var
prefix on a function call. This had NO EFFECT: the function read a
script-level variable that had already been evaluated once, long before
the self-test branch ran, so the prefixed env var was never consulted.
Running `--self-test` caught this immediately and unambiguously
("synthetic failure was NOT detected"). Fixed by making the threshold an
explicit function argument instead of an environment side-channel.
Notably, this bug was caught PRECISELY BECAUSE the script's failure path
was itself being tested (the entire point of `--self-test`, matching
`fpga_engine/symbiyosys`'s and `adversarial/`'s "prove the failure path
works, not just the happy path" discipline) -- a script whose only test
was "does the happy path print HEALTHY" would never have surfaced this.

## 4. What this phase does and doesn't claim

This phase claims: real, complete, correct implementations for every
step PLAN.md's build order specifies, each one genuinely run wherever
running it needed nothing beyond what's already on this Mac (steps 4-8,
10's read-only half, and step 2's WorkStealingPool half all fall into
this category), and honestly, specifically toolchain-gated -- with the
EXACT toolchain gap empirically characterized rather than assumed --
everywhere it doesn't (steps 1, 2's OpenMP half, 3, 9, 10's live-tuning
half).

It does NOT claim: that Slurm's real backfill/fairshare scheduling
behavior, MPI's real ring-vs-`MPI_Allreduce` performance gap, or
OpenMP's real parallel speedup have been measured -- those numbers stay
`TODO: run on [hardware]` in each step's own README until real Linux
cluster hardware (or, for steps 1-2, just the declined-this-session
`open-mpi`/`libomp` install) closes the gap.

## 5. Cross-reference

See `hpc_cluster/README.md` for the per-step status table and
`READING_LIST.md`'s Phase 22 section for the full citation list, and its
"Adjacent Engineering Disciplines" appendix (item 5) for step 11's
rack/facilities reference material.
