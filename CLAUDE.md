# Cross-Machine Runtime — Claude Context

Read PLAN.md and SCOPE.md at the start of every session before doing anything.

## Current status
- Phase 1, step 1 (CMake skeleton) — done
- Phase 1, step 2 (TSC benchmarking harness, `foundation/bench/bench.h`) — done
- Phase 1, step 3 (SPSC ring buffer, `foundation/spsc_queue.h`) — done. TSan clean. Benchmarks: 10 ns roundtrip (1T), 106 M items/sec throughput, 44 ns ping-pong RTT (release).
- Phase 1, step 4 (MPMC ring buffer, `foundation/mpmc_queue.h`) — done. TSan clean. Benchmarks: 21 ns roundtrip (1T), 52 M items/sec 1P-1C, 29 M 2P-2C, 12 M 4P-4C (release). 2x overhead vs SPSC in 1P-1C due to CAS.
- Phase 1, step 5 (ABA problem, `foundation/aba/`) — done. BuggyStack demo + AbaStack with 16-byte tagged-pointer CAS (cmpxchg16b). 25 ns (8B CAS) vs 39 ns (16B CAS). TSan clean.
- Phase 1, step 6 (hazard pointers, `foundation/hazard/`) — done. HazardDomain (global retire list, seq_cst slots, domain ID for tl() safety) + HazardStack<T>. 214 ns roundtrip. TSan clean.
- Phase 1, step 7 (epoch-based reclamation, `foundation/epoch/`) — done. EpochDomain (3-slot epoch cycling, global retire list) + EpochStack<T> (plain CAS — EBR prevents ABA implicitly). 171 ns roundtrip. TSan clean. Design doc in header: when to choose EBR vs hazard pointers.
- Next: Phase 1, step 8 — RCU (`foundation/rcu/`)

## Tooling decisions
- **Compiler:** Apple clang 14. No pre-built LLVM binaries exist for Intel macOS — brew always builds from source (2-5 hrs). Apple clang handles C++20/23 and all sanitizers fine for Phase 1.
- **clang-tidy:** Deferred. Will be set up when LLVM is built from source in Phase 4 step 1 (that build is intentional — it's a learning exercise).
- **Build system:** Ninja. CMake generates Ninja files.
- **Presets:** debug, release, asan, tsan, ubsan — each builds to its own `build/<preset>/` directory. Run with `cmake --preset <name>` and `cmake --build --preset <name>`.

## Execution plan (agreed 2026-05-31)
1. Finish Phase 1 fully (steps 5–18) with real implementations and tests.
2. Scaffold all phases 2–12 with real sample implementations — actual CUDA kernels, Raft, ZeRO, GBDT, etc. Hardware-gated with #ifdef/find_package where toolchain absent. Written as a safety checkpoint; user will walk through code with Claude to understand it.
3. Work through phases one by one to build real understanding.

## Phase 12: Machine Learning (added 2026-05-31)
Inserted after Phase 6 (distributed training), before Phase 9 (inference serving). See PLAN.md for full detail.
- **12a — Classical ML:** Random Forest, GBT (XGBoost/LightGBM-style), SVM, k-NN, k-means++, PCA
- **12b — Decision Framework + Benchmarks:** cross-method ablation on OpenML CC-18, written decision criteria, ensemble composition rules, hyperparameter sensitivity analysis
- **12c — Hyperparameter optimization:** Bayesian opt (GP), TPE, Hyperband/ASHA
- Benchmark targets: beat LightGBM throughput on large tabular while matching accuracy; compete on MLPerf
- Design docs are first-class: each algorithm family gets decision criteria, math foundations, failure modes

## Non-negotiable standards (from PLAN.md)
- Every component is benchmarked before moving on. No "I'll benchmark later."
- Every data structure has property-based tests verifying linearizability.
- TSan must find zero races on all lock-free data structures.
- Every non-obvious decision gets a written design doc before implementation.
- Hardware counter data (IPC, cache miss rates) on every CPU benchmark.
