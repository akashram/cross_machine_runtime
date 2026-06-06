# Cross-Machine Runtime — Claude Context

Read PLAN.md and SCOPE.md at the start of every session before doing anything.

## Where we are

**Phase 1: Foundation — COMPLETE (18/18 steps, 2026-06-02)**
All lock-free data structures, allocators, coroutine engine, tensor handle,
property-based testing framework, and hardware counter infrastructure are done.
Every component: TSan clean, zero warnings, benchmarked.
See git log for per-step detail; PLAN.md for the full spec.

**Next: Phase 2, step 1 — CPU affinity + thread pinning**
Directory: `cpu_engine/` (new top-level directory, sibling of `foundation/`)

---

## Phase 2: CPU Backend — kickoff notes

Phase 2 lives in `cpu_engine/`. Create it with its own CMakeLists.txt and
add `add_subdirectory(cpu_engine)` to the root CMakeLists.txt.

### Platform reality
- **Development:** macOS (Intel, Apple clang 14). Fine for steps 1–6.
- **AVX-512 (steps 4–9):** macOS Intel does NOT have AVX-512. Need a Linux
  cloud instance (AWS c5.2xlarge = Xeon Platinum 8275CL, AVX-512 supported).
  Use `#ifdef __AVX512F__` guards so steps build on macOS without AVX-512.
- **perf tool (steps 11+):** Linux only (`perf stat`, `perf record`).
  The `foundation/perf/` PerfCounters from step 18 handles this already.
- **hugepages (step 2):** `MAP_HUGETLB` = Linux only; `foundation/arena/`
  already has the pattern — copy it.

### Build order (from PLAN.md)
1. **CPU affinity + thread pinning** — `pthread_setaffinity_np` wrapper,
   pin thread to a core, measure scheduling jitter (TSC stddev) with/without.
   macOS: use `thread_policy_set` hint (advisory, not hard). Linux: hard pin.
   **THIS IS THE NEXT STEP.**

2. **Hugepage allocator** — 2 MB pages via `mmap(MAP_HUGETLB)` + `mbind`.
   Benchmark TLB miss reduction vs 4 KB pages. Already have the pattern in
   `foundation/arena/arena.h` and `foundation/numa/numa.h` — build on that.

3. **OS-level tuning scripts** — `isolcpus`, `nohz_full`, IRQ affinity, C-state
   disabling, CPU governor scripts. Linux only. Packaged as shell scripts with
   before/after latency measurements. macOS: document why not applicable.

4. **Non-temporal store primitives** — `_mm_stream_*` wrappers (SSE/AVX),
   benchmark write-only vs regular stores. Guards for non-x86.

5. **Prefetch primitives** — `__builtin_prefetch` wrappers (T0/T1/T2/NTA),
   measure prefetch distance vs cache miss rate via PerfCounters.

6. **Branchless primitives** — `cmov` patterns, branchless min/max/abs/clamp.
   Verify with branch miss rate from PerfCounters.

7. **AVX-512 kernel library** — dot product, matrix-vector multiply,
   element-wise ops, INT8 ops. NEEDS LINUX/CLOUD. Scalar baseline + 
   compiler-vectorized comparison always included for macOS CI.

8. **Cache-aware tiling** — blocked matmul, tune tile sizes to L1/L2.
   PerfCounters-driven measurement at each tile size.

9. **CPU inference engine** — small MLP through AVX-512 kernels, preallocated
   buffers, no heap on hot path.

10. **Roofline model** — peak FLOPS micro-benchmark, STREAM bandwidth,
    plot achieved vs ceiling, classify compute-bound vs bandwidth-bound.

11. **Hardware perf counter deep dive** — IPC, L1/L2/L3 miss rates, TLB misses
    per kernel. Uses `foundation/perf/PerfCounters`.

12. **PGO** — instrument build, representative workload, `-fprofile-use`.

13. **Busy-poll vs OS-wait** — p50/p99/p999 comparison.

### Step 1 design sketch (CPU affinity)
```
cpu_engine/
  affinity/
    affinity.h       — ThreadPinner class, pin_to_cpu(), current_cpu()
  test/
    CMakeLists.txt
    affinity_test.cpp
  CMakeLists.txt
```

`ThreadPinner`: wraps `pthread_setaffinity_np` (Linux) / `thread_policy_set`
(macOS). Exposes `pin(cpu_id)`, `unpin()`, `current_cpu()`.

Benchmark: measure TSC jitter (stddev of inter-sample deltas) on a pinned
vs unpinned thread. Expected: pinned ≈ 10–50 ns stddev; unpinned ≈ 200–2000 ns
(cache migration, branch predictor warm-up loss).

---

## Tooling decisions
- **Compiler:** Apple clang 14. No pre-built LLVM for Intel macOS.
- **clang-tidy:** Deferred to Phase 4 step 1 (LLVM source build).
- **Build system:** Ninja. CMake presets: debug/release/asan/tsan/ubsan.
  Run: `cmake --preset <name>` then `cmake --build --preset <name>`.
- **AVX-512:** Requires `--preset release` on a Linux AVX-512 machine.
  Add a new `avx512` preset when we reach step 7.

## Non-negotiable standards
- Every component is benchmarked before moving on.
- Property-based tests for every data structure.
- TSan zero races on all concurrent code.
- Every non-obvious decision gets a written design doc before implementation.
- Hardware counter data (IPC, cache miss rates) on every CPU benchmark.
- Every step gets a `README.md` in its component directory written after
  seeing the measured numbers. Document: what was built, key results table,
  findings/interpretation, platform notes. See `cpu_engine/tiling/README.md`
  through `cpu_engine/perf_deep_dive/README.md` for the established format.
- Commit AND push to origin/main after every completed step.

## Execution plan (agreed 2026-05-31, updated 2026-06-02)
Phase 1 done. Moving directly into Phase 2 step by step (skipping the
"scaffold all phases" checkpoint — user wants to work through them directly).
