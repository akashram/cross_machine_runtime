# Cross-Machine Runtime — Claude Context

Read PLAN.md and SCOPE.md at the start of every session before doing anything.

## Current status
- Phase 1, step 1 (CMake skeleton) — done
- Phase 1, step 2 (TSC benchmarking harness, `foundation/bench/bench.h`) — done
- Phase 1, step 3 (SPSC ring buffer, `foundation/spsc_queue.h`) — done. TSan clean. Benchmarks: 11 ns roundtrip (1 thread), 20 ns/item throughput, 43 ns ping-pong RTT (release, 2.3 GHz Mac).
- Next: Phase 1, step 4 — MPMC ring buffer (`foundation/mpmc_queue.h`)

## Tooling decisions
- **Compiler:** Apple clang 14. No pre-built LLVM binaries exist for Intel macOS — brew always builds from source (2-5 hrs). Apple clang handles C++20/23 and all sanitizers fine for Phase 1.
- **clang-tidy:** Deferred. Will be set up when LLVM is built from source in Phase 4 step 1 (that build is intentional — it's a learning exercise).
- **Build system:** Ninja. CMake generates Ninja files.
- **Presets:** debug, release, asan, tsan, ubsan — each builds to its own `build/<preset>/` directory. Run with `cmake --preset <name>` and `cmake --build --preset <name>`.

## Non-negotiable standards (from PLAN.md)
- Every component is benchmarked before moving on. No "I'll benchmark later."
- Every data structure has property-based tests verifying linearizability.
- TSan must find zero races on all lock-free data structures.
- Every non-obvious decision gets a written design doc before implementation.
- Hardware counter data (IPC, cache miss rates) on every CPU benchmark.
