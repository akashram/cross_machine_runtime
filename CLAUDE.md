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

**Phase 3: GPU Backend — CODE COMPLETE (24/24 steps, 2026-07-19)**
All steps implemented with real CUDA code: memory management, streams, warp
and shared-memory primitives, coalescing, occupancy, elementwise/GEMM kernels,
PTX/SASS inspection, flash attention, CUDA graphs, P2P, mixed precision, FP8,
tensor core alignment, Hopper TMA/WGMMA, 2:4 sparsity, roofline, MPS, NVML
power monitoring, Nsight CI. Lives in `gpu_engine/`. None of it has run on a
GPU yet — no CUDA toolchain on Mac. All README.md result tables are still
`TODO: run on [hardware]`. Hardware validation is deferred (see below).

**Phase 4: Compiler/IR (MLIR) — CODE COMPLETE (15/15 steps, 2026-07-19)**
Runtime dialect (15 ops, 3 attrs, TableGen-based) plus all nine passes:
shape inference, fusion, affine lowering/tiling, memory planning, remat,
placement, auto-sharding, kernel specialization, and the AOT pipeline that
orchestrates all of them + LLVM codegen + link. Lives in `compiler/`. None
of it has run — no MLIR/LLVM toolchain on Mac — except `cost_model/`,
which has no MLIR dependency, compiles with plain `clang++`, and has
actually been run locally (see `compiler/cost_model/README.md` for
captured output). See `compiler/DESIGN.md` for the design rationale.

**Phase 5: Distributed Layer + Networking — CODE COMPLETE (25/25 steps, 2026-07-19)**
Shared portable `common/Channel` transport (real POSIX sockets) plus every
step: EFA/RDMA (code-complete, hardware-gated), PTP, gRPC control plane,
FlatBuffers data plane, AF_XDP/userspace networking, NIC deep dive,
ring/halving-doubling/tree all-reduce, the broadcast/reduce-scatter/
all-gather library, NCCL tuning config, topology-aware scheduler, vector
clocks, Chandy-Lamport snapshots, Raft consensus (leader election + log
replication), TLA+ specs for both Raft and the collective protocol,
backpressure/hedged requests/multi-tenancy, and a chaos engineering
harness. Lives in `networking/`. Unlike Phases 3–4, **12 of the 25 steps
are actually built and run locally** (Mac, real sockets/threads, zero
EFA/Linux dependency) — `common`, `rdma_v1`'s TCP baseline, `efa_srd`,
`ring_allreduce`, `halving_doubling`, `tree_allreduce`, `collectives`,
`topo_scheduler`, `vector_clocks`, `chandy_lamport`, `raft`,
`backpressure`, `hedged_requests`, `multitenancy` all have real captured
test output in their READMEs (`ctest` in the repo root runs all of them:
39/39 passing). See `networking/DESIGN.md` for the design rationale and
several real bugs caught by actually running the code (a ring-algorithm
chunk-ownership off-by-one, two shutdown-coordination deadlocks). The
remaining 13 steps are real, complete code gated behind Linux-only kernel
APIs, a specific NIC, external libraries, GPU hardware, or a Java
toolchain for TLC — see `networking/README.md`'s status table.

**Phase 6: Distributed GPU Training — IN PROGRESS (1/25 steps, started 2026-07-21)**
Step 1 (distributed data loading) is code-complete and locally run: a
WebDataset-style tar codec, shard-granularity rank sharding, and a
multi-worker prefetch queue, all portable (no CUDA/Linux dependency) —
built and tested on this Mac (`ctest -R data_loader_test`, real captured
numbers in `distributed_training/data_loading/README.md`). Remaining 24
steps are still stubs. Next: step 2 (GPUDirect Storage).

**Phases 7, 8, 9, 10, 12 — STUBBED, pending full local implementation**
Stub directories, interface headers, CMakeLists.txt, and README.md design
outlines exist. Code bodies are still 1–3 line TODOs. Next up after Phase 6,
in PLAN.md order: Phase 7 (FPGA Backend).

---

## Execution strategy (updated 2026-07-19)

**Write every phase's real code on the Mac (or any hardware-free tooling)
before spending a dollar on cloud hardware. No cloud instance gets
provisioned until every phase below is code-complete.**

This reverses the earlier "stub first, fill in on hardware" approach: stubs
turned out too bare to be useful as a plan (a few lines of declarations per
component), so the actual algorithms get written now, locally, and cloud
hardware is used purely for benchmarking/tuning what already works on paper.

**Local implementation order: PLAN.md phase order (1 → 2 → 3 → 4 → 5 → 6 → 7
→ 8 → 9 → 10 → 12).** Phases 1–3 are done (Phase 3 is code-complete, not yet
hardware-validated). Next is Phase 4, then 5, 6, 7, 8, 9, 10, 12 in sequence.
Within each phase, implement step by step in PLAN.md's build order. A step
counts as implemented when it has real logic (not a stub) and compiles
wherever it can without the target hardware; benchmark numbers stay TODO
until the hardware validation pass.

**Hardware validation pass (after all phases above are code-complete):**
Work through phases in the same order, one hardware type at a time.

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
2. `git pull origin/main` to get the full implementation.
3. Build with the appropriate CMake preset for that platform.
4. Run and tune each step in PLAN.md order within that phase.
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
