# openmp_port -- OpenMP reduction vs. WorkStealingPool

**Status: code-complete; the `foundation::WorkStealingPool` half is
locally run today; the real-OpenMP half is toolchain-gated (no `libomp`
installed -- ask-before-install declined this session).**

## What this measures

PLAN.md Phase 22 step 2: port a hand-threaded kernel to OpenMP pragmas,
compared directly against the hand-rolled thread-pool version for
correctness and measured speedup. Kernel: parallel reduction (sum) over a
20M-element float array -- the textbook case OpenMP's `reduction()`
clause exists for, and a fair point of comparison against
`foundation::WorkStealingPool::parallel_for` (`foundation/ws_pool/ws_pool.h`),
which has **no built-in reduction primitive** at all: `openmp_reduce.cpp`
hand-rolls the same "shard, reduce locally, combine" shape `reduction(+:sum)`
gives OpenMP for free.

## A real platform gotcha, caught by writing this, not by reading docs

Apple clang does **not** ship `libomp`. `#pragma omp parallel for
reduction(+:sum)` compiles cleanly without `-fopenmp` -- an unrecognized
`#pragma` is legal, silently-ignored C++, not a build error. That means
an "OpenMP" build on a machine without the runtime **degrades silently to
a serial for-loop**, not a compile failure -- a real correctness/reporting
trap: anyone who assumes a missing `-fopenmp` flag would be caught at
build time is wrong. `openmp_reduce.cpp` checks `_OPENMP` (only defined
by the compiler when a real OpenMP implementation is active) at runtime
and labels its own output honestly rather than silently reporting a
"parallel" time that's actually the serial fallback.

## Toolchain gate

No `libomp` is installed on this Mac. This session ran `brew info libomp`
and confirmed a real, current, bottled Homebrew formula exists (23.1.1,
keg-only -- not symlinked into `/usr/local` by default since it can
override GCC headers) -- so, like step 1's `open-mpi`, this step's gate is
the standing no-new-local-installs decision for this session, not a
genuine platform limitation.

`openmp_reduce` is built unconditionally (unlike `mpi_ring_allreduce`,
which hard-requires `<mpi.h>` to even compile): the `WorkStealingPool`
half of the comparison is real, portable, and runs today; only the omp
pragma's *parallel* behavior needs `libomp` linked via
`OpenMP::OpenMP_CXX` (added by `hpc_cluster/CMakeLists.txt` when
`find_package(OpenMP)` succeeds).

## Results (captured 2026-09-12, Apple clang 14, this Mac, `--preset release`, `libomp` NOT linked)

```
openmp_reduce: N=20000000, threads=4, shards=16
  serial baseline              : sum=4309.379187  25.571 ms
  foundation::WorkStealingPool : sum=4309.379187  PASS  8.240 ms  (3.10x vs serial)
  #pragma omp parallel for     : sum=4309.379187  PASS  24.634 ms  (1.04x vs serial)  [_OPENMP NOT defined -> SERIAL FALLBACK (no libomp linked)]
  NOTE: this build has no libomp -- the omp pragma above compiled as a
  no-op and ran serially. Its timing is NOT a real OpenMP speedup
  measurement; see README for the toolchain gate.
```

Stable across 3 repeated runs: `WorkStealingPool` speedup 2.99x-3.22x,
serial-fallback omp path 0.99x-1.05x (i.e. genuinely ~1.0x, as it must be
for identical serial code) -- see `hpc_cluster/README.md`'s phase-level
table for the full captured log.

## A real debug-vs-release measurement bug, caught by running this, not by reading the code

The first captured run used the `debug` preset (`-O0`, per
`CMakePresets.json` -- no `-O2`/`-O3`) and reported a nonsensical
**3.66x "speedup" for the serial-fallback omp path** (`_OPENMP` NOT
defined, i.e. genuinely single-threaded C++) over the plain serial
baseline, and a `WorkStealingPool` speedup of **7.39x on only 4 hardware
threads** -- super-linear, which a shard-then-combine reduction with no
superlinear-cache effect has no mechanism to produce honestly. Root
cause (per this repo's standing "root-cause a suspicious measurement
structurally, don't paper over it" discipline): `serial_sum` originally
used a range-based `for (float v : data)` loop while `omp_parallel_sum`
used an indexed `for (int i = 0; i < n; ++i)` loop -- at `-O0`, with no
inlining or vectorization to erase the difference, the two loop shapes
have measurably different per-iteration overhead purely from
`std::vector<float>::iterator` dereference cost vs. `operator[]`, which
had nothing to do with OpenMP, threading, or cache warmth. Fixed by
making `serial_sum` use the identical indexed-loop shape as
`omp_parallel_sum` (see that function's own comment) and re-measuring
under `--preset release` (`-O2`+): the serial-fallback omp path now
correctly reports ~1.0x (it IS the same serial code), and
`WorkStealingPool`'s 4-thread speedup lands at a believable ~3.0-3.2x
(sub-linear, as expected -- pool dispatch overhead and the kernel being
memory-bandwidth-bound on an 80MB array, not compute-bound, are the
standard reasons a 4-thread reduction doesn't hit a clean 4.0x). This is
exactly the kind of measurement artifact `-O0` debug builds can hide
inside an otherwise-correct program -- worth a benchmark always running
under `release`, not just debug, before trusting a "speedup" number.

## Findings

- **Correctness holds for both paths** on this Mac: `WorkStealingPool`'s
  shard-then-combine sum and the (serially-executing) omp-pragma sum both
  match the serial ground truth to `1e-3` relative tolerance -- floating-
  point summation order changes the exact bit pattern (that's expected;
  see `distributed_training`'s own gradient-summation tolerance
  conventions), not the value at this tolerance.
- **`WorkStealingPool` gets a real, believable 3.0-3.2x speedup on 4
  hardware threads** for this reduction, once measured correctly under
  `release` -- sub-linear (not 4.0x) because the kernel is memory-
  bandwidth-bound on an 80MB array (20M `float`s) rather than
  compute-bound, and because of real thread-pool dispatch overhead
  (`parallel_for`'s `TaskGroup` construction/teardown per call).
- **The real comparison this step is meant to answer -- does
  `reduction()`'s compiler-managed shape actually beat a hand-rolled
  thread pool's reduction on real parallel hardware -- is still unrun**,
  since only the serial-fallback path exists without `libomp`. TODO once
  `libomp` is linked: rerun and fill in a real parallel-OpenMP speedup
  number to compare against `WorkStealingPool`'s real 3.0-3.2x.

## Hardware/toolchain notes

- Required for a real (non-serial-fallback) OpenMP measurement:
  `brew install libomp`, then re-configure with `-fopenmp` and
  `-L$(brew --prefix libomp)/lib -I$(brew --prefix libomp)/include`
  (CMake's `find_package(OpenMP)` handles this automatically once
  `libomp` is discoverable -- Apple clang needs `OpenMP_CXX_FLAGS`
  pointed at the keg-only install location; see CMake's own
  `FindOpenMP` module docs for the Apple-clang-specific hint variables
  if auto-detection doesn't find it).
- Run: `./build/debug/hpc_cluster/openmp_port/openmp_reduce`.
