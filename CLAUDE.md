# Cross-Machine Runtime — Claude Context

Read PLAN.md and SCOPE.md at the start of every session before doing anything.

## Where we are

**Phase 1: Foundation — COMPLETE (18/18 steps, 2026-06-02)**
All lock-free data structures, allocators, coroutine engine, tensor handle,
property-based testing framework, and hardware counter infrastructure are done.
Every component: TSan clean, zero warnings, benchmarked.

**Phase 2: CPU Backend — COMPLETE (13/13 steps)**
All CPU affinity, hugepages, OS tuning, SIMD, branchless, AVX-512, tiling,
inference engine, roofline, perf counters, PGO, and busy-poll steps done.
Lives in `cpu_engine/`.

**Phase 3: GPU Backend — IN PROGRESS (5/24 steps done)**
Steps 1–5 implemented (CMake scaffold, GPU memory, streams, warp primitives,
shared memory). Steps 6–24 are stubs: directory + interface + README with
measurement TODOs. Lives in `gpu_engine/`.

**Phases 4–10, 12 — STUBBED (2026-06-20)**
All remaining phases have stub directories, interface headers, CMakeLists.txt,
and README.md files with design outlines. Nothing has been run on hardware yet.
Full cloud hardware validation is the next major milestone after stubbing.

---

## Execution strategy (updated 2026-06-20)

**Stub-first, then validate on cloud hardware.**

All phase directories, API headers, benchmark skeletons, and README outlines
exist in the repo. The code compiles (or is gated behind hardware checks).
Real benchmark numbers are TODO throughout stubs.

**When cloud hardware is available, work through stubs in phase order:**

### Hardware needed per phase
| Phase | Hardware | AWS instance |
|---|---|---|
| Phase 3 (GPU) | NVIDIA GPU, CUDA | g4dn.xlarge → p3.2xlarge → p4d.24xlarge |
| Phase 4 (MLIR) | Linux (compile LLVM from source) | any Linux x86 |
| Phase 5 (Distributed) | Multi-node + EFA | 2× p4d.24xlarge in placement group |
| Phase 6 (Distributed Training) | Multi-GPU | p4d.24xlarge |
| Phase 7 (FPGA) | Xilinx UltraScale+ | F1 spot (~$0.50/hr) |
| Phase 8 (TPU) | Google TPU | GCP v4-8 or TRC |
| Phase 9 (Inference) | GPU with large VRAM | p3.2xlarge or p4d |
| Phase 10 (Observability) | Linux (eBPF) | any Linux |
| Phase 12 (ML) | Any (mostly CPU) | c5.2xlarge |

### When returning to a phase on cloud hardware
1. SSH into the appropriate instance.
2. `git pull origin/main` to get all stubs.
3. Build with the appropriate CMake preset for that platform.
4. Work through stub directories in PLAN.md order within that phase.
5. Fill in README.md results tables with real numbers.
6. Commit and push after each step.

---

## Tooling decisions
- **Compiler:** Apple clang 14 on Mac. GCC/clang on Linux cloud instances.
- **clang-tidy:** Deferred to Phase 4 step 1 (LLVM source build on Linux).
- **Build system:** Ninja. CMake presets: debug/release/asan/tsan/ubsan.
  Run: `cmake --preset <name>` then `cmake --build --preset <name>`.
- **AVX-512:** Requires `--preset release` on a Linux AVX-512 machine.
- **CUDA:** GPU stubs build only when `CMAKE_CUDA_COMPILER` is detected.
  Root CMakeLists.txt gates `add_subdirectory(gpu_engine)` via check_language(CUDA).
- **MLIR/LLVM:** Build from source on Linux. Deferred entirely to Phase 4.
- **FPGA (Vitis):** TCL-driven headless Vivado on AWS F1 AMI.

## Non-negotiable standards
- Every component is benchmarked before moving on.
- Property-based tests for every data structure.
- TSan zero races on all concurrent code.
- Every non-obvious decision gets a written design doc before implementation.
- Hardware counter data (IPC, cache miss rates) on every CPU benchmark.
- Every step gets a `README.md` in its component directory written after
  seeing the measured numbers. Document: what was built, key results table,
  findings/interpretation, platform notes.
- Commit AND push to origin/main after every completed step.
- **Stubs:** README.md files have a `## Results` section marked `TODO: run on [hardware]`.
  Fill these in with real numbers when validating on cloud hardware.
