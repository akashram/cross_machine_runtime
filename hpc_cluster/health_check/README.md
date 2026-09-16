# health_check -- real node health-check tooling, wired as Slurm's `HealthCheckProgram`

**Status: complete and run locally today, both the healthy path and the
failure path. Real bug caught and fixed while writing this -- see below.**

## What this measures

PLAN.md Phase 22 step 4: a real node health-check script in the shape of
Slurm's own `HealthCheckProgram` mechanism, wired as
`slurm_jobs/slurm.conf`'s actual `HealthCheckProgram=`/`HealthCheckInterval=300`
directives -- not a standalone script disconnected from step 3. Per
Slurm's documented semantics, `slurmd` runs this on every `IDLE`/`MIXED`
node on that interval, and a non-zero exit automatically `DRAIN`s the
node. Checks what's actually checkable locally regardless of whether
Slurm itself runs here: disk space, expected process/service presence,
and basic hardware sanity.

## Design

Three checks, each degrading honestly rather than silently:
1. **Disk space** (`/` and, if present, `/scratch` -- the real directory
   `slurm_jobs/train_job.sbatch` writes checkpoint shards under):
   configurable free-space threshold, default 10%.
2. **Expected process presence** (`slurmd`, `munged`): on Linux (the real
   target), a missing process is a genuine failure. On this Mac, where
   neither is installed at all, the check reports `SKIP` rather than a
   false `FAIL` -- this Mac was never going to run `slurmd`, so treating
   its absence as an unhealthy-node signal would be meaningless noise,
   not a real finding.
3. **CPU count sanity**: confirms the node reports at least the CPU
   count `slurm_jobs/slurm.conf`'s `NodeName=` expects -- catches a node
   that silently lost a core (BIOS/firmware fault) before it accepts
   jobs sized for a core count it no longer has.

`--self-test` injects a synthetic, guaranteed-to-fail disk-space
threshold (101%, impossible to satisfy) to prove the FAILURE path -- and
therefore the exit-code contract `HealthCheckProgram` depends on --
actually works, without needing a genuinely unhealthy node to test
against. Same "prove the failure path, not just the happy path"
discipline as `fpga_engine/symbiyosys`'s k-induction proofs and
`adversarial/`'s attack-verification steps.

## A real bug caught by running `--self-test`, not by inspection

The first version read the disk-space threshold from a script-level
variable (`MIN_FREE_PCT`, set once from `$HEALTH_CHECK_MIN_FREE_PCT` at
parse time) directly inside `check_disk()`. The self-test tried to
inject a guaranteed-to-fail threshold by setting the env var as a
one-shot prefix on the function call:
`HEALTH_CHECK_MIN_FREE_PCT=101 check_disk "/"`. That has **no effect**:
`MIN_FREE_PCT` was already evaluated once, long before the self-test
branch ever runs -- the prefixed env var never gets read again. Running
`--self-test` caught this immediately and unambiguously:

```
--- self-test: injecting a synthetic disk-space failure ---
OK: disk /: 95% free (threshold 10%)
self-test FAIL: synthetic failure was NOT detected -- health check logic is broken
```

Fixed by making the threshold an explicit second argument to
`check_disk()` (`check_disk "/" 101`) instead of an environment
side-channel read inside the function. Re-run confirmed the fix:

```
--- self-test: injecting a synthetic disk-space failure ---
UNHEALTHY: disk /: only 95% free (threshold 101%)
self-test PASS: synthetic failure correctly detected and would DRAIN the node
```

## Results (captured 2026-09-16, this Mac)

**Self-test (failure path):**
```
--- self-test: injecting a synthetic disk-space failure ---
UNHEALTHY: disk /: only 95% free (threshold 101%)
self-test PASS: synthetic failure correctly detected and would DRAIN the node
exit code: 0
```

**Normal run (healthy path):**
```
node_health_check.sh: Akashs-MacBook-Pro.local, 2026-09-16T10:41:00Z
OK: disk /: 95% free (threshold 10%)
SKIP: disk check on /scratch (not present)
SKIP: process 'slurmd' not found (expected -- slurmd is Linux/Slurm-only, this is macOS)
SKIP: process 'munged' not found (expected -- munged is Linux/Slurm-only, this is macOS)
OK: CPU count 4 (>= expected minimum 1)
node_health_check.sh: HEALTHY
exit code: 0
```

## Findings

- **Both the healthy and unhealthy paths are genuinely exercised on this
  Mac today** -- unlike most of this phase's other steps, this one needs
  no Linux/Slurm toolchain to validate its own core logic (the exit-code
  contract `HealthCheckProgram` depends on), only to validate the
  Linux-specific checks (`slurmd`/`munged` presence) it correctly `SKIP`s
  here.
- **A genuinely macOS-vs-Linux-honest design decision paid off during
  debugging**: because the script reports `SKIP` (not `FAIL`) for
  Linux-only checks on this Mac, the self-test's real bug was isolated
  to exactly one check (`check_disk`) rather than buried in a wall of
  expected `slurmd`/`munged` failures that would have been noise on this
  platform.

## Hardware/toolchain notes
- Portable checks (disk, CPU count) run on any POSIX system, verified on
  this Mac today.
- The `slurmd`/`munged` process checks only become load-bearing (able to
  genuinely FAIL) on real Linux Slurm cluster hardware -- same gate as
  `slurm_jobs/` itself.
