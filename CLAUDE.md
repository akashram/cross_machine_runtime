# Cross-Machine Runtime — Claude Context

Read PLAN.md and SCOPE.md at the start of every session before doing anything.

## Current status
Phase 1, step 1 (CMake skeleton) complete. Next: Phase 1 step 2 — TSC-based benchmarking harness in `foundation/`.

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
