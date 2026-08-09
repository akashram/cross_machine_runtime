# Cross-Machine Runtime — Reading List

Everything needed to understand this repo as deeply as the assistant working on
it does: the in-repo docs to read (and in what order), the external papers
those docs cite or build on, the vendor/spec manuals the hardware-gated code
targets, and the commit(s) that actually built each step. PLAN.md and
SCOPE.md describe *what to build*; this document is the *background +
citation + provenance* layer underneath them, assembled by reading every
README.md/DESIGN.md, grepping every source comment in the repo for actual
citations, and matching every step against `git log` (this repo commits
almost exactly one step per commit, with "Phase N step M" in the message —
see Cross-cutting below).

**Structure:** split by phase, then by individual PLAN.md build-order step —
every step in every phase gets its own line, even the ones with no external
citation (marked explicitly as "no dedicated citation," not silently
omitted) and even the ones with no commit yet (marked "not yet implemented"),
so this doubles as a completeness check against PLAN.md's build order.
Nothing from the earlier version of this document was dropped in the
split — every citation below appeared in that version too, just
consolidated under a phase instead of a step.

## How to use this

- Read in the order below — it's PLAN.md's phase order, which is also the
  dependency order (Phase 6 reuses Phase 5's collectives, Phase 9 reuses
  Phase 6's transformer, Phase 13 reuses Phase 6+9, etc.).
- For each phase: read the "Start here" in-repo docs first, then walk the
  step list — most steps have no external citation at all (they're API
  mechanics, glue code, or measurement/analysis steps), so treat "no
  dedicated citation" as a signal to move on, not a gap to fill.
- **The in-repo docs are the primary source.** Every step's own README.md
  documents what was actually built and what was actually measured (or, for
  hardware-gated steps, what's still TODO) — the papers below are the
  background those READMEs assume, not a replacement for reading them.
- **Each step's commit hash is a `git show <hash>` away from the exact diff
  that built it** — useful when a README describes the end state but you
  want to see the change in isolation, or when a step was built across more
  than one commit (a primary "Implement ..." commit plus a later fix,
  listed together).
- The final two sections are flat indexes (alphabetical bibliography; vendor
  docs) for looking a specific citation up without walking the whole phase
  structure.
- **Phases 17-19** (added 2026-08-09, scoped in PLAN.md/SCOPE.md/CLAUDE.md)
  close a gap found by comparing this repo against a batch of analog/
  physics-based-AI-compute job descriptions. Phases 17 and 18 are both
  CODE COMPLETE (8/8 and 10/10 steps) as of 2026-08-09; Phase 19 hasn't
  started. Its steps have no in-repo doc pointer or commit yet — treat
  them as pure background reading to have in hand before that code lands.

---

## Tier 0: Prerequisites (read before Phase 1)

The whole repo assumes working knowledge of:

- **C++23** — the entire codebase's language. Concepts, `std::atomic`,
  structured bindings, ranges. (User is learning this on the job — see
  project memory; no specific book mandated, cppreference is the day-to-day
  reference used throughout.)
- **The C++ memory model** — every lock-free structure in `foundation/`
  depends on acquire/release/seq-cst semantics being right. Herb Sutter's
  "atomic<> Weapons" talks (CppCon) and Preshing on Programming's blog
  ("Acquire and Release Semantics", "The Happens-Before Relation") are the
  standard on-ramp; `foundation/DESIGN.md` (see Phase 1 below) documents the
  specific ordering decisions made in this repo.
- **Computer architecture fundamentals** — cache hierarchy, coherence
  (MESI), false sharing, NUMA, branch prediction, out-of-order execution.
  Needed for Phase 2 (CPU) and referenced throughout Phases 3/7/8/15 as the
  baseline every accelerator is compared against.
- **Linear algebra basics** (matrix multiply, SVD, eigendecomposition) —
  needed for Phase 4 (tiling/affine lowering), Phase 12 (PCA/kernels), and
  every GEMM-shaped kernel in Phases 2/3/7/8.
- **Probability/statistics basics** (confidence intervals, hypothesis
  testing) — needed for Phase 14 step 8 (randomized smoothing's
  Clopper-Pearson vs. Wald bound) and Phase 12b/12c's benchmark
  methodology.

---

## Phase 1: Foundation — lock-free data structures, allocators, coroutines

**Start here:** `foundation/DESIGN.md`, each subdirectory's own README.md.

- **Step 1 — CMake project skeleton** [`c40606e`]: no dedicated citation (build-system mechanics).
- **Step 2 — Statistical benchmarking harness** [`f46c6f2`]: no dedicated citation (TSC/RDTSC timing mechanics).
- **Step 3 — SPSC ring buffer** [`cb26480`]: no dedicated citation.
- **Step 4 — MPMC ring buffer** [`442c508`]: no dedicated citation (CAS-based array queue; distinct from step 10's linked-list Michael-Scott queue).
- **Step 5 — ABA problem** (`foundation/aba/`) [`d35934d`]: no dedicated citation in-repo; see the Herlihy & Shavit background below for the standard treatment.
- **Step 6 — Hazard pointers** (`foundation/hazard/`) [`d002e7e`]: Michael, M. (2004), *"Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects"*.
- **Step 7 — Epoch-based reclamation** (`foundation/epoch/`) [`303ca2e`]: no dedicated citation in-repo; see the Herlihy & Shavit background below.
- **Step 8 — RCU (Read-Copy-Update)** (`foundation/rcu/`) [`1c67d08`]: Desnoyers, M. et al. (2012), *"User-Level Implementations of Read-Copy Update"* — `foundation/rcu/rcu_domain.h` cites this directly ("based on URCU, Desnoyers et al. 2012") for the per-thread-counter algorithm.
- **Step 9 — Lock-free freelist** (`foundation/freelist/`) [`22c32fd`]: no separate citation — built on step 6's hazard pointers for safe reclamation.
- **Step 10 — Lock-free queue (Michael-Scott)** (`foundation/mpmc_queue.h`) [`47e4ddc`]: Michael, M. & Scott, M. (1996), *"Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms"*.
- **Step 11 — Chase-Lev work-stealing deque** (`foundation/chase_lev/`) [`a2cf8b1`, fix: `a1b26e9`]: Chase, D. & Lev, Y. (2005), *"Dynamic Circular Work-Stealing Deque"* — `chase_lev.h` implements this directly.
- **Step 12 — Work-stealing thread pool** (`foundation/ws_pool/`) [`c1f2b6f`]: no separate citation — built directly on step 11's deque.
- **Step 13 — Coroutine execution engine** (`foundation/coro/`) [`a632e08`]: Baker, L., cppcoro (CppCon 2019 talk) — `coro.h` cites this directly for the `AsyncMutex` design.
- **Step 14 — Arena allocator** (`foundation/arena/`) [`8dd615d`]: no dedicated citation.
- **Step 15 — NUMA-aware allocator** [`bdf7aff`]: no dedicated citation.
- **Step 16 — Unified tensor handle (v1)** (`foundation/tensor/`) [`aa82809`]: no dedicated citation.
- **Step 17 — Property-based testing setup** (`foundation/proptest*`) [`05cd0ec`]: Claessen, K. & Hughes, J. (2000), *"QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs"* — the paradigm `foundation/proptest` implements in C++.
- **Step 18 — x86 hardware counter infrastructure** (`foundation/perf/`) [`5e3a278`]: no dedicated citation (uses `perf_event_open()` directly; see the Intel manual under Phase 2 for the counters themselves).

**Background (applies across steps 5-11):** Herlihy, M. & Shavit, N., *The
Art of Multiprocessor Programming* — the standard textbook covering ABA,
hazard pointers, epoch reclamation, and work-stealing in one place; useful
connective tissue between the papers above and for the steps that have no
dedicated citation of their own.

---

## Phase 2: CPU Backend — affinity, SIMD, tiling, inference engine

**Start here:** `cpu_engine/DESIGN.md`.

- **Step 1 — CPU affinity + thread pinning** (`affinity/`) [`08b3991`]: no dedicated citation.
- **Step 2 — Hugepage allocator** (`hugepage/`) [`0ba690e`]: Drepper, U. (2007), *"What Every Programmer Should Know About Memory"*.
- **Step 3 — OS-level tuning scripts** (`os_tuning/`) [`ed641f7`]: no dedicated citation.
- **Step 4 — Non-temporal store primitives** (`nt_store/`) [`886a63f`]: no dedicated citation.
- **Step 5 — Prefetch primitives** (`prefetch/`) [`e05dc47`]: Drepper, U. (2007) (same as step 2).
- **Step 6 — Branchless primitives** (`branchless/`) [`ad8d53d`]: no dedicated citation.
- **Step 7 — AVX-512 kernel library** (`avx512/`) [`7404592`]: Intel 64 and IA-32 Architectures Optimization Reference Manual — instruction latencies/throughput.
- **Step 8 — Cache-aware tiling** (`tiling/`) [`4335616`]: no dedicated citation.
- **Step 9 — CPU inference engine** (`inference/`) [`08389e1`]: no dedicated citation.
- **Step 10 — Roofline model (CPU)** (`roofline/`) [`11494e7`]: Williams, S., Waterman, A. & Patterson, D. (2009), *"Roofline: An Insightful Visual Performance Model for Multicore Architectures"* — cited directly by `roofline.h`; the standard "compute-bound vs. memory-bound" lens reused in Phases 3 and 8.
- **Step 11 — Hardware perf counter deep dive** (`perf_deep_dive/`) [`79ad79b`, docs follow-up: `1144e69`]: no dedicated citation.
- **Step 12 — Profile-guided optimization (PGO)** (`pgo/`) [`b6a9f0c`]: no dedicated citation; assumes familiarity with profile-guided compilation (`-fprofile-generate`/`-fprofile-use`).
- **Step 13 — Busy-poll vs OS-wait comparison** (`busy_poll/`) [`8a5a066`]: no dedicated citation; assumes familiarity with the latency/CPU-utilization tradeoff vs. blocking syscalls (`epoll`/`futex`).

---

## Phase 3: GPU Backend — CUDA (code-complete, hardware-gated)

**Start here:** `gpu_engine/DESIGN.md`. None of this phase has run on real
hardware (no CUDA toolchain on Mac) — every README's `## Results` is
`TODO: run on [hardware]`.

- **Step 1 — CUDA project integration** [`9860245`]: no dedicated citation; see *CUDA C++ Programming Guide* under Vendor docs.
- **Step 2 — GPU memory management** [`4edc7a2`]: no dedicated citation.
- **Step 3 — Stream manager** [`d7b97ee`]: no dedicated citation.
- **Step 4 — Warp-level primitives library** (`warp_primitives/`) [`e7ee5a2`]: no dedicated citation.
- **Step 5 — Shared memory primitives** [`d99c55e`]: no dedicated citation.
- **Step 6 — Memory coalescing validator** (`coalescing/`) [`91d851c`]: no dedicated citation.
- **Step 7 — Occupancy tuner** (`occupancy/`) [`91d851c`]: no dedicated citation.
- **Step 8 — Elementwise GPU kernels** (`kernels/`) [`91d851c`]: no dedicated citation.
- **Step 9 — GEMM kernel** (`kernels/`) [`91d851c`]: no dedicated citation.
- **Step 10 — PTX/SASS inspection workflow** (`ptx_sass/`) [`91d851c`]: no dedicated citation; see *PTX ISA* under Vendor docs.
- **Step 11 — Flash Attention forward kernel** (`flash_attn/`) [`91d851c`]: Dao, T., Fu, D., Ermon, S., Rudra, A. & Ré, C. (2022), *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"* — cited directly by `flash_attn/README.md`; Milakov, M. & Gimelshein, N. (2018), *"Online normalizer calculation for softmax"* — the running-max/normalizer trick, also cited directly, used inside the kernel.
- **Step 12 — Flash Attention backward kernel** (`flash_attn/`) [`91d851c`]: Dao et al. (2022), same paper — covers the recomputation-based backward pass too.
- **Step 13 — CUDA Graphs** (`graphs/`) [`91d851c`]: no dedicated citation.
- **Step 14 — GPUDirect P2P** (`p2p/`) [`91d851c`]: no dedicated citation.
- **Step 15 — Mixed precision** (`precision/`) [`91d851c`]: Micikevicius, P. et al. (2017), *"Mixed Precision Training"*.
- **Step 16 — FP8 (Hopper)** (`precision/`/`hopper/`) [`91d851c`]: no separate citation beyond step 15's mixed-precision background.
- **Step 17 — Tensor Core alignment analysis** (`precision/`) [`91d851c`]: no dedicated citation.
- **Step 18 — Hopper TMA** (`hopper/`) [`91d851c`]: NVIDIA, *Hopper Architecture Whitepaper*.
- **Step 19 — Hopper WGMMA** (`hopper/`) [`91d851c`]: NVIDIA, *Hopper Architecture Whitepaper* (same as step 18).
- **Step 20 — 2:4 structured sparsity** (`sparsity/`) [`91d851c`]: no dedicated citation (implementation target is `cusparseLtMatmul`).
- **Step 21 — Roofline model (GPU)** (`roofline/`) [`91d851c`]: Williams, Waterman & Patterson (2009), same as Phase 2 step 10.
- **Step 22 — CUDA MPS setup** (`mps/`) [`91d851c`]: no dedicated citation.
- **Step 23 — NVML power monitoring** (`power/`) [`91d851c`]: no dedicated citation.
- **Step 24 — Nsight integration in CI** (`nsight_ci/`) [`91d851c`]: no dedicated citation.
- **Step 25 — Triton kernels** (added 2026-08-09): **not yet implemented** (scoped in `e523a33`); no dedicated citation once built — see Triton language docs under Vendor docs.
- **Step 26 — CUTLASS GEMM** (added 2026-08-09): **not yet implemented** (scoped in `e523a33`); no dedicated citation once built — see CUTLASS docs under Vendor docs.

Steps 6-24 all landed in one commit, `91d851c` ("Implement Phase 3 GPU
backend steps 6–24, all stubs → full CUDA code"), which replaced the
earlier stub pass (`b80bd2a`) with real code for all 19 steps at once —
that's why they share a hash above; `git show 91d851c -- gpu_engine/<dir>`
narrows to one step's diff.

**Vendor docs (map to the steps above):** *CUDA C++ Programming Guide*
(steps 1-3, 6-7, 13-14), *Parallel Thread Execution (PTX) ISA* (step 10),
Nsight Compute/Systems docs (steps 6-7, 24), NVML API reference (step 23) —
all needed once this phase is validated on real hardware.

---

## Phase 4: Compiler / IR (MLIR) — code-complete, hardware-gated

**Start here:** `compiler/DESIGN.md`, `compiler/README.md`. Only
`compiler/cost_model/` has actually run locally (no MLIR dependency, plain
`clang++`); everything else needs an LLVM/MLIR toolchain build, deferred to
Linux hardware validation.

All 15 steps landed in one commit, `34cbd9a` ("Implement Phase 4
compiler/IR (MLIR) steps 1-15: all stubs -> full code"), following the same
earlier stub pass (`b80bd2a`) as Phase 3's steps 6-24. `git show 34cbd9a --
compiler/<dir>` narrows to one step's diff.

- **Step 1 — MLIR build setup** [`34cbd9a`]: no dedicated citation; read Lattner et al. (2021) below before starting this step.
- **Step 2 — Runtime dialect design** (`dialect/`) [`34cbd9a`]: Lattner, C. et al. (2021), *"MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"* (CGO 2021) — the foundational paper for the whole phase's IR design (TableGen ops, dialect conversion, pass pipelines).
- **Step 3 — Dialect registration + parsing** [`34cbd9a`]: Lattner et al. (2021), same paper (covers TableGen ODS directly).
- **Step 4 — Shape inference pass** (`shape_inference/`) [`34cbd9a`]: no dedicated citation.
- **Step 5 — Operator fusion pass** (`fusion/`) [`34cbd9a`]: no dedicated citation.
- **Step 6 — Affine dialect lowering** (`affine_lower/`) [`34cbd9a`]: no dedicated citation beyond Lattner et al. (2021)'s coverage of MLIR's existing Affine dialect.
- **Step 7 — Memory planning pass** (`mem_planning/`) [`34cbd9a`]: no dedicated citation.
- **Step 8 — Rematerialization pass** (`remat/`) [`34cbd9a`]: no dedicated citation; draws on the classic register-allocation-by-graph-coloring/spill-cost literature (Chaitin et al.) as background.
- **Step 9 — Device placement pass** (`placement/`) [`34cbd9a`]: no dedicated citation.
- **Step 10 — Auto-sharding pass** (`sharding/`) [`34cbd9a`]: no dedicated citation; parallels GSPMD/Megatron-style sharding annotations — see Phase 6 step 11's Megatron-LM citation for the same idea applied at the model-code level instead of the compiler-IR level.
- **Step 11 — Kernel specialization** (`kernel_spec/`) [`34cbd9a`]: no dedicated citation.
- **Step 12 — AOT compilation pipeline** (`aot/`) [`34cbd9a`]: Lattner, C. & Adve, V. (2004), *"LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation"* (CGO 2004) — LLVM codegen background for the pipeline's backend.
- **Step 13 — Cost model** (`cost_model/`) [`34cbd9a`]: no dedicated citation.
- **Step 14 — libFuzzer integration** (`fuzzing/`) [`34cbd9a`]: no dedicated citation; see libFuzzer docs under Vendor docs.
- **Step 15 — LLVM upstream** [`34cbd9a`]: Lattner & Adve (2004), same as step 12.

**Vendor docs:** MLIR language reference, TableGen ODS reference, LLVM IR
reference manual.

---

## Phase 5: Distributed Layer + Networking

**Start here:** `networking/DESIGN.md`, `networking/README.md` (has the full
per-step status table: 12 of 25 steps actually built+run on this Mac).

All 25 steps landed in one commit, `db0e54d` ("Implement Phase 5 networking
steps 1-25: all stubs -> full code"). `git show db0e54d -- networking/<dir>`
narrows to one step's diff.

- **Step 1 — EFA setup and validation** (`efa_setup/`) [`db0e54d`]: no dedicated citation; see libfabric/EFA guide under Vendor docs.
- **Step 2 — libfabric RDMA transport (v1)** (`rdma_v1/`) [`db0e54d`]: no dedicated citation.
- **Step 3 — One-sided RDMA operations** (`rdma_onesided/`) [`db0e54d`]: no dedicated citation.
- **Step 4 — EFA SRD transport** (`efa_srd/`) [`db0e54d`]: no dedicated citation.
- **Step 5 — PTP clock synchronization** (`ptp/`) [`db0e54d`]: IEEE 1588 (Precision Time Protocol) standard.
- **Step 6 — gRPC + protobuf control plane** (`grpc_control/`) [`db0e54d`]: no dedicated citation; see gRPC C++ docs under Vendor docs.
- **Step 7 — Flatbuffers data plane** (`flatbuffers_data/`) [`db0e54d`]: no dedicated citation; see FlatBuffers docs under Vendor docs.
- **Step 8 — AF_XDP kernel bypass** (`af_xdp/`) [`db0e54d`]: no dedicated citation; see AF_XDP kernel docs under Vendor docs.
- **Step 9 — Userspace networking stack** (`userspace_net/`) [`db0e54d`]: no dedicated citation.
- **Step 10 — NIC hardware deep dive** (`nic_deep_dive/`) [`db0e54d`]: no dedicated citation.
- **Step 11 — Ring all-reduce** (`ring_allreduce/`) [`db0e54d`]: Patarasuk, P. & Yuan, X. (2009), *"Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations"* — background for the bandwidth-optimality claim.
- **Step 12 — Recursive halving-doubling all-reduce** (`halving_doubling/`) [`db0e54d`]: Rabenseifner, R. (2004), *"Optimization of Collective Reduction Operations"* — `halving_doubling/README.md` names this directly ("Rabenseifner's algorithm").
- **Step 13 — Tree all-reduce** (`tree_allreduce/`) [`db0e54d`]: no dedicated citation.
- **Step 14 — Broadcast, reduce-scatter, all-gather** (`collectives/`) [`db0e54d`]: no dedicated citation.
- **Step 15 — NCCL integration + tuning** (`nccl_tuning/`) [`db0e54d`]: no dedicated citation; see NCCL developer guide under Vendor docs.
- **Step 16 — Topology-aware scheduler** (`topo_scheduler/`) [`db0e54d`]: no dedicated citation.
- **Step 17 — Vector clocks** (`vector_clocks/`) [`db0e54d`]: Lamport, L. (1978), *"Time, Clocks, and the Ordering of Events in a Distributed System"* — the scalar `LamportClock`; Fidge/Mattern (independently, 1988) — the vector-clock extension `VectorClock` implements.
- **Step 18 — Chandy-Lamport snapshots** (`chandy_lamport/`) [`db0e54d`]: Chandy, K.M. & Lamport, L. (1985), *"Distributed Snapshots: Determining Global States of Distributed Systems"* — cited directly ("the classic algorithm, Chandy & Lamport, 1985").
- **Step 19 — Raft consensus** (`raft/`) [`db0e54d`, real bug fix: `6cca717` "Fix raft_test segfault: real use-after-free, not a timing flake"]: Ongaro, D. & Ousterhout, J. (2014), *"In Search of an Understandable Consensus Algorithm"* — election timeout range (150-300ms) taken directly from the paper.
- **Step 20 — TLA+ spec for Raft** (`tla_raft/`) [`db0e54d`]: Ongaro, D. (2014), *Consensus: Bridging Theory and Practice* (Stanford PhD thesis) — `tla_raft/README.md` cites the thesis appendix directly as the source the community TLA+ Raft spec follows; Lamport, L. (2002), *Specifying Systems* — the TLA+ book needed to read `Raft.tla`.
- **Step 21 — Backpressure + load shedding** (`backpressure/`) [`db0e54d`]: no dedicated citation.
- **Step 22 — Hedged requests** (`hedged_requests/`) [`db0e54d`]: no dedicated citation.
- **Step 23 — Multi-tenancy** (`multitenancy/`) [`db0e54d`]: no dedicated citation.
- **Step 24 — Chaos engineering harness** (`chaos/`) [`db0e54d`]: no dedicated citation.
- **Step 25 — TLA+ for collective protocol** (`tla_collective/`) [`db0e54d`]: Lamport (2002), *Specifying Systems*, same as step 20.

**Vendor docs:** libfabric/EFA programmer's guide, gRPC C++ docs,
FlatBuffers schema/codegen docs, AF_XDP kernel documentation, NCCL
developer guide.

---

## Phase 6: Distributed GPU Training (+ `/transformer/`)

**Start here:** `distributed_training/README.md`, plus `transformer/README.md`.

**The model itself:** [`63c89de`] "Add minimal decoder-only transformer +
tokenizer" — Vaswani, A. et al. (2017), *"Attention Is All You Need"* — the
architecture `transformer/transformer_model.h` and
`tensor_parallel_attn/attention.h` implement; read this before any step
below that trains or manipulates the model. Built between steps 21 and 22
specifically because 22-25 need a real model to train.

- **Step 1 — Distributed data loading** (`data_loading/`) [`990dd77`]: no dedicated citation.
- **Step 2 — GPUDirect Storage** (`gpudirect_storage/`) [`88392a5`]: no dedicated citation.
- **Step 3 — Data parallel training (baseline)** (`data_parallel/`) [`0547c82`]: no dedicated citation.
- **Step 4 — Gradient accumulation** (`grad_accum/`) [`923d486`]: no dedicated citation.
- **Step 5 — Gradient clipping** (`grad_clipping/`) [`5712fae`]: no dedicated citation.
- **Step 6 — Autograd engine** (`autograd/`) [`3f7cc53`]: no dedicated citation; standard backprop/optimizer material (SGD, Adam) is the assumed background.
- **Step 7 — ZeRO-1** (`zero1/`) [`e437763`]: Rajbhandari, S. et al. (2020), *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"*.
- **Step 8 — ZeRO-2** (`zero2/`) [`8eb3a27`]: Rajbhandari et al. (2020), same paper.
- **Step 9 — ZeRO-3** (`zero3/`) [`0ac6df7`]: Rajbhandari et al. (2020), same paper.
- **Step 10 — ZeRO-Infinity** (`zero_infinity/`) [`63dc4c9`]: Rajbhandari, S. et al. (2021), *"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning"*.
- **Step 11 — Column/row-parallel linear layers** (`col_row_linear/`) [`75169f6`]: Shoeybi, M. et al. (2019), *"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"* — cited directly as the pattern being validated.
- **Step 12 — Tensor-parallel attention** (`tensor_parallel_attn/`) [`07a6738`]: Shoeybi et al. (2019), same paper; Vaswani et al. (2017) for the attention mechanism itself.
- **Step 13 — Sequence parallelism** (`seq_parallel/`) [`42f7f9b`]: Korthikanti, V. et al. (2022), *"Reducing Activation Recomputation in Large Transformer Models"*.
- **Step 14 — 1F1B pipeline schedule** (`pipeline_1f1b/`) [`d22194a`]: Huang, Y. et al. (2019), *"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"* — the baseline `pipeline_schedule.h` compares against directly; Narayanan, D. et al. (2019 PipeDream / 2021 Megatron-LM pipelining paper) — cited as "see the Megatron-LM paper" for the bubble-time equivalence result.
- **Step 15 — 3D parallelism** (`parallel_3d/`) [`a53c977`]: no separate citation — composition of steps 7-14's already-cited techniques.
- **Step 16 — MoE / Expert parallelism** (`moe/`) [`0b0df78`]: Shazeer, N. et al. (2017), *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"*; Fedus, W., Zoph, B. & Shazeer, N. (2021), *"Switch Transformers"*.
- **Step 17 — Checkpoint sharding** (`checkpoint/`) [`eeda1ff`]: no dedicated citation.
- **Step 18 — Compute/communication overlap** (`compute_comm_overlap/`) [`9467362`]: no dedicated citation.
- **Step 19 — Distributed batch normalization** (`sync_batchnorm/`) [`7e6a09f`]: Ioffe, S. & Szegedy, C. (2015), *"Batch Normalization"*; Ioffe, S. (2017), *"Batch Renormalization"*.
- **Step 20 — Full training loop** (`full_training_loop/`) [`8013592`]: no dedicated citation.
- **Step 21 — 2:4 sparsity in training** (`sparsity_training/`) [`dab6989`]: no separate citation — cross-references Phase 3 step 20's `cusparseLtMatmul` target.
- **Step 22 — Supervised fine-tuning (SFT)** (`sft/`) [`15a7439`]: Ouyang, L. et al. (2022), *"Training language models to follow instructions with human feedback"* (InstructGPT/RLHF) — the SFT→reward-model→PPO pipeline this step starts.
- **Step 23 — Reward model training** (`reward_model/`) [`fb8da20`]: Bradley, R.A. & Terry, M.E. (1952), *"Rank Analysis of Incomplete Block Designs"* — the pairwise-preference model `reward_model.h` builds on directly.
- **Step 24 — PPO-based RLHF** (`ppo_rlhf/`) [`f9dc5ce`]: Schulman, J. et al. (2017), *"Proximal Policy Optimization Algorithms"* — the clipped surrogate objective and "k1" KL estimator, cited directly; Christiano, P. et al. (2017), *"Deep Reinforcement Learning from Human Preferences"*; Stiennon, N. et al. (2020), *"Learning to Summarize from Human Feedback"*; Ouyang et al. (2022), same as step 22.
- **Step 25 — DPO (Direct Preference Optimization)** (`dpo/`) [`b5abad4`]: Rafailov, R. et al. (2023), *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"* — cited directly; the step is built on their closed-form-optimal-policy reparameterization.
- **Step 26 — io_uring-based async checkpoint I/O**: **not yet implemented** (scoped only: `5d836cb` "Scope io_uring checkpoint I/O as Phase 6 step 26"); no dedicated citation once built — see `liburing`/io_uring docs under Vendor docs (Linux-only).

**Background:** policy-gradient/REINFORCE (Sutton & Barto, *Reinforcement
Learning: An Introduction*, ch. 13) is useful background before step 24's
Schulman et al. PPO paper.

---

## Phase 7: FPGA Backend — code-complete, hardware-gated

**Start here:** `fpga_engine/README.md` (per-step status table).

- **Step 1 — AWS F1 setup** (`f1_setup/`) [`5db5a85`]: no dedicated citation.
- **Step 2 — TCL build pipeline** (`tcl_pipeline/`) [`a7039dd`]: no dedicated citation; see Vivado UG901/UG904 under Vendor docs.
- **Step 3 — Vivado power report CI** (`power_ci/`) [`9d30ed1`]: no dedicated citation.
- **Step 4 — AXI4-Stream interface** (`axi_stream/`) [`e54add2`]: no dedicated citation.
- **Step 5 — First HLS kernel: vector dot product** (`dot_product/`) [`e54add2`]: no dedicated citation; see Vitis HLS UG1399 under Vendor docs. (Steps 4-5 landed together: "Implement Phase 7 steps 4-5: AXI4-Stream passthrough + dot product II study".)
- **Step 6 — Loop optimization deep dive** (`loop_opt/`) [`e7f6479`]: no dedicated citation; see Vitis HLS UG1399.
- **Step 7 — DSP vs LUT tradeoff analysis** (`dsp_lut/`) [`9e528e0`]: no dedicated citation; see UltraScale+ architecture docs under Vendor docs. Background: any intro RTL/FPGA design text (e.g. Harris & Harris, *Digital Design and Computer Architecture*) covers the pipeline-register/critical-path vocabulary this step assumes.
- **Step 8 — Fixed-point arithmetic** (`fixed_point/`) [`7a927a7`]: no dedicated citation; see Vitis HLS UG1399 (`ap_fixed`).
- **Step 9 — BRAM vs URAM access patterns** (`bram_uram/`) [`b526df0`]: no dedicated citation; see UltraScale+ architecture docs.
- **Step 10 — DDR4 integration** (`ddr4/`) [`4596154`]: no dedicated citation.
- **Step 11 — DMA orchestration** (`dma/`) [`8405865`]: no dedicated citation; see XRT Architecture and API under Vendor docs.
- **Step 12 — PCIe latency decomposition** (`pcie_latency/`) [`b436abd`]: no dedicated citation.
- **Step 13 — Ping-pong double buffering** (`pingpong/`) [`7a276fc`]: no dedicated citation.
- **Step 14 — ML inference kernel** (`ml_kernel/`) [`97bf822`]: no dedicated citation.
- **Step 15 — Timing closure** (`timing_closure/`) [`2576e4a`]: no dedicated citation; same Harris & Harris background as step 7.
- **Step 16 — SLR partitioning** (`slr/`) [`f501eec`]: no dedicated citation; see UltraScale+ architecture docs.
- **Step 17 — Clock gating** (`clock_gating/`) [`4d44c47`]: no dedicated citation.
- **Step 18 — XADC monitoring** (`xadc/`) [`2392f35`]: no dedicated citation; see XRT Architecture and API (`get_info<thermal|electrical>()`).
- **Step 19 — ILA debug session** (`ila_debug/`) [`319af74`]: no dedicated citation.
- **Step 20 — Cocotb testbenches** (`cocotb/`) [`fe6e444`]: no dedicated citation; see cocotb documentation under Vendor docs.
- **Step 21 — SymbiYosys formal verification** (`symbiyosys/`) [`cfbdd28`, actually run: `67271f0` "Run Phase 7 step 21 SymbiYosys formal proofs, both PASS by k-induction"]: no academic citation; see SymbiYosys/Yosys documentation under Vendor docs — the README documents a real gap (free `yosys` lacks full SVA `assert property` grammar, needs the commercial Verific plugin) that forced rewriting properties into procedural `assert`/`$past`/`$stable` form.
- **Step 22 — Partial reconfiguration** (`partial_reconfig/`) [`59179b8`]: no dedicated citation.
- **Step 23 — FPGA network stack** (`fpga_net/`) [`00a86f7`]: no academic citation; see P4_16 Language Specification under Vendor docs.
- **Step 24 — Vitis AI evaluation** (`vitis_ai/`) [`5c47bd0`]: no academic citation; see Vitis AI User Guide under Vendor docs (DPU architecture, `vai_q_pytorch` quantization, `vai_c_xir` compilation).
- **Step 25 — Thermal-aware router integration** (`thermal_router/`) [`10617b6`, follow-up: `321e190` "Add poll-interval sweep to thermal_router_sim, empirically check the claim"]: no dedicated citation.

**Vendor docs:** Xilinx/AMD *Vivado Design Suite User Guide* (UG901/UG904),
*Vitis HLS User Guide* (UG1399), *UltraScale+ Architecture* documents, *XRT
Architecture and API*, *Vitis AI User Guide*; P4 Language Consortium,
*P4_16 Language Specification*; SymbiYosys/Yosys documentation; cocotb
documentation.

---

## Phase 8: TPU Backend — code-complete, hardware-gated

**Start here:** `tpu_engine/README.md` (per-step status table).

- **Step 1 — GCP + TPU setup** (`gcp_setup/`) [`cdcb300`]: no dedicated citation.
- **Step 2 — TPU hardware deep dive** (`tpu_benchmarks/`) [`21fb57b`]: Jouppi, N. et al. (2017/2023), *"In-Datacenter Performance Analysis of a Tensor Processing Unit"* and the TPU v4 follow-up paper.
- **Step 3 — MLIR → StableHLO lowering** (`stablehlo_lower/`) [`1101ead`]: StableHLO project (OpenXLA), *StableHLO specification* — the target dialect for `StableHLOLowerPass.cpp`.
- **Step 4 — StableHLO → XLA execution** (`stablehlo_execute/`) [`5089fb8`]: Google, *XLA: Optimizing Compiler for Machine Learning* documentation; Google/JAX docs (`jax.export`).
- **Step 5 — TPU memory layout optimization** (`layout_opt/`) [`f898840`]: Jouppi et al. (2017/2023), same as step 2 — the systolic-array MXU architecture this step's tile-padding model reasons about.
- **Step 6 — Explicit HBM ↔ SRAM scheduling** (`hbm_sram/`) [`7677475`]: Jouppi et al. (2017/2023), same as step 2.
- **Step 7 — pjit for distributed TPU** (`pjit_distributed/`) [`3b3df21`]: Google/JAX docs (`pjit`, sharding annotations).
- **Step 8 — Multi-TPU collectives via ICI** (`ici_collectives/`) [`0a6c4da`]: no dedicated citation.
- **Step 9 — MXU utilization optimization** (`mxu_opt/`) [`723d8ea`]: Jouppi et al. (2017/2023), same as step 2.
- **Step 10 — VLIW ISA analysis** (`vliw_analysis/`) [`2c43409`]: Jouppi et al. (2017/2023); assumes the same computer-architecture background as Phase 2/3 (instruction scheduling, hazard hiding) for the VLIW-vs-OOO-vs-SIMT comparison — no additional paper needed.
- **Step 11 — TPU Profiler integration** (`tpu_profiler/`) [`1bda536`]: no dedicated citation.
- **Step 12 — TPU vs GPU cost model** (`cost_model/`) [`6367429`]: no dedicated citation.
- **Step 13 — SparseCore (TPU v5)** (`sparsecore/`) [`14ffdc9`]: no dedicated citation.

---

## Phase 9: Inference Serving

**Start here:** `inference_serving/README.md` (per-step status table; most
steps actually run locally).

- **Step 1 — Paged KV cache** (`paged_kv/`) [`e612f8a`]: Kwon, W. et al. (2023), *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* (vLLM paper) — the block-table design implemented directly.
- **Step 2 — Continuous batching** (`continuous_batching/`) [`2d15cf0`]: Kwon et al. (2023), same paper — the continuous-batching throughput finding reproduced as a real measurement.
- **Step 3 — SLA-aware scheduler** (`sla_scheduler/`) [`25fab21`]: no dedicated citation.
- **Step 4 — FlashDecoding** (`flash_decoding/`) [`217edb5`]: Dao, T. et al. (2023), *"Flash-Decoding for long-context inference"* — the split-K-over-KV-dimension idea this step targets.
- **Step 5 — Speculative decoding** (`speculative_decoding/`) [`2474aaf`]: Leviathan, Y., Kalman, M. & Matias, Y. (2023), *"Fast Inference from Transformers via Speculative Decoding"* (and the contemporaneous Chen et al. 2023 DeepMind paper).
- **Step 6 — GPTQ INT4 quantization** (`gptq/`) [`76f6791`]: Frantar, E. et al. (2022), *"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"* — cited directly (Hessian inversion + error compensation onto not-yet-quantized columns).
- **Step 7 — KV cache quantization** (`kv_quant/`) [`9c1ab56`]: no dedicated citation.
- **Step 8 — Backend-agnostic serving** (`serving_backend/`) [`ab362c6`]: no dedicated citation.
- **Step 9 — Serving benchmarks** (`serving_bench/`) [`783143e`]: no dedicated citation.

---

## Phase 10: Observability, Testing, AI Integration

**Start here:** `observability/README.md`.

- **Step 1 — eBPF probes** (`ebpf/`) [`7f11c8f`]: Gregg, B., *BPF Performance Tools* — the standard reference for tracepoints/kprobes used in `sched_probe.py`.
- **Step 2 — OpenTelemetry integration** (`opentelemetry/`) [`a19b6a4`]: Sigelman, B. et al. (2010), *"Dapper, a Large-Scale Distributed Systems Tracing Infrastructure"* — the span/parent-child model followed; OpenTelemetry project, *OTLP specification* — the exact JSON export format parsed back and validated against.
- **Step 3 — Unified observability dashboard** (`dashboard/`) [`d140db1`]: no dedicated citation.
- **Step 4 — TLC model checking** (`tlc/`) [`fd68a1d`]: Lamport, L. (2002), *Specifying Systems* — TLC usage, same book as Phase 5 steps 20/25.
- **Step 5 — SymbiYosys CI integration** (`symbiyosys_ci/`) [`6b004a8`]: no dedicated citation — re-runs Phase 7 step 21's proofs.
- **Step 6 — Chaos test suite** (`chaos/`) [`b30841c`]: no dedicated citation.
- **Step 7 — Nsight profile analyzer agent** (`nsight_agent/`) [`9b8d2f9`]: no academic citation; see Anthropic API reference under Vendor docs.
- **Step 8 — Kernel variant generator agent** (`kernel_variant_agent/`) [`5f51d7a`]: no academic citation; see Anthropic API reference.
- **Step 9 — LLM autotuning agent** (`llm_autotune/`) [`044fc6c`]: no academic citation; see Anthropic API reference.
- **Step 10 — Tool use for the existing agents** (added 2026-07-28): **not yet implemented** (scoped only, in SCOPE.md's AI Integration section); no academic citation once built — see Anthropic API reference (Messages API tool-use mechanism).
- **Step 11 — Skill registry + composable dispatch** (added 2026-07-28): **not yet implemented** (scoped only); no dedicated citation.

**Vendor docs:** Anthropic API reference (Messages API, structured
outputs/`output_config.format`) — the client code in steps 7-9 targets
this directly.

---

## Phase 12: Machine Learning Library

**Start here:** `ml/README.md`.

### Phase 12a: Classical ML (steps 1-8)
- **Step 1 — Decision tree (CART)** (`decision_tree/`) [`65f46d2`, follow-up: `90baa5f` "decision_tree: add real per-split feature subsampling for random_forest"]: Breiman, L. et al. (1984), *Classification and Regression Trees*.
- **Step 2 — Random Forest** (`random_forest/`) [`a6ea5cc`]: Breiman, L. (2001), *"Random Forests"*.
- **Step 3 — Gradient Boosted Trees** (`gradient_boosting/`) [`796dfbb`]: Friedman, J. (2001), *"Greedy Function Approximation: A Gradient Boosting Machine"*; Friedman, J. (2002), *"Stochastic Gradient Boosting"* (row subsampling).
- **Step 4 — SVM** (`svm/`) [`6b289d4`]: Platt, J. (1998), *"Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines"*.
- **Step 5 — k-NN** (`knn/`) [`9e1a1ec`]: Bentley, J.L. (1975), *"Multidimensional Binary Search Trees"* (kd-tree, branch-and-bound); Omohundro, S. (1989), *"Five Balltree Construction Algorithms"*; Uhlmann, J. (1991) (ball trees, independently); Liu, T. et al. (2004) (defeatist-search approximate ball-tree ANN).
- **Step 6 — k-means++** (`kmeans/`) [`f6c7eeb`]: Arthur, D. & Vassilvitskii, S. (2007), *"k-means++: The Advantages of Careful Seeding"*.
- **Step 7 — PCA** (`pca/`) [`1919aaa`]: Halko, N., Martinsson, P.G. & Tropp, J. (2011), *"Finding Structure with Randomness"* (randomized SVD).
- **Step 8 — Linear models** (`linear_models/`) [`0f653d2`]: Zou, H. & Hastie, T. (2005), *"Regularization and Variable Selection via the Elastic Net"*; Nocedal, J. & Wright, S., *Numerical Optimization*, Algorithm 7.4/7.5 (L-BFGS).

### Phase 12b: Decision Framework + Benchmarks (steps 9-14)
- **Step 9 — OpenML CC-18 runner** (`openml_bench/`) [`94bddaf`]: Bischl, B. et al. (2019), *"OpenML Benchmarking Suites"* — `study/99` is the actual OpenML-CC18 study ID.
- **Step 10 — Cross-method comparison** (`cross_method/`) [`06c2099`]: no dedicated citation — grounded in step 9's real numbers.
- **Step 11 — Decision criteria document** (`decision_criteria/`) [`03390e6`]: no dedicated citation.
- **Step 12 — Hyperparameter sensitivity analysis** (`hyperparam_sensitivity/`) [`f0300bc`]: no dedicated citation.
- **Step 13 — Ensemble composition** (`ensemble/`) [`e4f22f0`]: no dedicated citation.
- **Step 14 — Failure mode catalog** (`failure_modes/`) [`2d5e081`]: no dedicated citation.

### Phase 12c: Hyperparameter Optimization (steps 15-18)
- **Step 15 — Bayesian optimization** (`bayesian_opt/`) [`e12948e`]: Rasmussen, C.E. & Williams, C.K.I. (2006), *Gaussian Processes for Machine Learning*; Mockus, J. (1978), *"The Application of Bayesian Methods for Seeking the Extremum"* (Expected Improvement).
- **Step 16 — TPE** (`tpe/`) [`6fdcbb9`]: Bergstra, J. et al. (2011), *"Algorithms for Hyper-Parameter Optimization"* (Tree-structured Parzen Estimator).
- **Step 17 — Hyperband / ASHA** (`hyperband/`) [`796961d`]: Jamieson, K. & Talwalkar, A. (2016), *"Non-stochastic Best Arm Identification and Hyperparameter Optimization"* (Successive Halving); Li, L. et al. (2016), *"Hyperband"*; Li, L. et al. (2018), *"Massively Parallel Hyperparameter Tuning"* (ASHA).
- **Step 18 — Population-based training** (`pbt/`) [`91f0bca`]: Jaderberg, M. et al. (2017), *"Population Based Training of Neural Networks"*.

**Background (steps 15-18 collectively):** Bergstra, J. & Bengio, Y.
(2012), *"Random Search for Hyper-Parameter Optimization"* — the
"random search is a strong baseline" finding all four steps independently
reproduce on this repo's real OpenML tasks.

**Phase-level background:** read Breiman (2001), Friedman (2001), and Platt
(1998) alongside their implementations (steps 2, 3, 4) — this phase
demonstrates ML systems depth, not a substitute for the papers themselves.

---

## Phase 13: Retrieval-Augmented Generation

**Start here:** `rag/DESIGN.md`.

- **Step 1 — Embedding model** (`embedding_model/`) [`b7c7056`]: van den Oord, A., Li, Y. & Vinyals, O. (2018), *"Representation Learning with Contrastive Predictive Coding"* — the InfoNCE loss cited directly; Radford, A. et al. (2021), *"Learning Transferable Visual Models From Natural Language Supervision"* (CLIP) — the symmetric query↔document contrastive formulation, also cited directly.
- **Step 2 — Cosine-similarity ANN retrieval** (`ml/knn`'s `CosineBallTree`) [`cb45828`]: no dedicated citation — extends the existing `BallTree`.
- **Step 3 — HNSW index** (`hnsw/`) [`52a54b0`]: Malkov, Y. & Yashunin, D. (2016/2018), *"Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"* — implements Algorithm 2 (SEARCH-LAYER) directly, cited by name.
- **Step 4 — Chunking + indexing pipeline** (`indexing_pipeline/`) [`7957e4f`]: no dedicated citation.
- **Step 5 — Retrieval-augmented prompt construction** (`rag_generation/`) [`8e4dd30`]: Lewis, P. et al. (2020), *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* — the retrieve-then-generate paradigm this step implements.
- **Step 6 — Recall@k measurement** (`recall_eval/`) [`9368ec0`]: no dedicated citation.
- **Step 7 — Generation quality with vs. without retrieval** (`generation_quality/`) [`ab55f97`]: no dedicated citation.
- **Step 8 — Approximate vs. exact retrieval, system-level** (`approx_retrieval_study/`) [`af5bba3`]: no dedicated citation.
- **Step 9 — Serving integration** (`serving_integration/`) [`cf46417`]: no dedicated citation.

---

## Phase 14: Adversarial Robustness

**Start here:** `adversarial/DESIGN.md`.

- **Step 1 — Input gradients** (`input_gradients/`) [`d98e3e8`]: no dedicated citation — foundational; verifies the existing autograd tape's `x.grad()` via finite differences.
- **Step 2 — FGSM** (`fgsm/`) [`ef0e1c8`]: Goodfellow, I., Shlens, J. & Szegedy, C. (2014), *"Explaining and Harnessing Adversarial Examples"*.
- **Step 3 — PGD** (`pgd/`) [`99fcd9c`]: Madry, A. et al. (2017), *"Towards Deep Learning Models Resistant to Adversarial Attacks"*.
- **Step 4 — Undefended vulnerability measurement** (`vulnerability_measurement/`) [`c7adcf7`]: no separate citation — composes steps 2-3 against the undefended model.
- **Step 5 — Adversarial training** (`adversarial_training/`) [`e3309e3`]: Madry et al. (2017), same paper — the min-max training formulation.
- **Step 6 — Robustness/accuracy tradeoff** (`robustness_tradeoff/`) [`e966553`]: Tsipras, D. et al. (2018), *"Robustness May Be at Odds with Accuracy"*.
- **Step 7 — Transferability study** (`transferability/`) [`3bfce2e`]: Szegedy, C. et al. (2013), *"Intriguing Properties of Neural Networks"*; Papernot, N. et al. (2016), *"Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples"*.
- **Step 8 — Randomized smoothing** (`randomized_smoothing/`) [`901e209`]: Cohen, J., Rosenfeld, E. & Kolter, Z. (2019), *"Certified Adversarial Robustness via Randomized Smoothing"*. Background: Clopper, C.J. & Pearson, E.S. (1934), *"The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial"* — the exact bound Cohen et al. use, which `randomized_smoothing.h` documents substituting with a normal/Wald approximation (a disclosed simplification).

---

## Phase 15: NPU Backend

**Start here:** `npu_engine/DESIGN.md`.

- **Step 1 — Toolchain validation**: **not yet implemented** (hardware-gated — no ANE/Coral hardware or `coremltools`/`onnx`/`onnxruntime` locally); no dedicated citation once built — see Apple Core ML / ONNX Runtime NPU execution providers docs under Vendor docs.
- **Step 2 — Quantization export pipeline** (`quant_export/`) [`71a1840`, test fix: `8676ed3` "Fix Phase 15 step 2 test: correct compression-ratio bound assumption"]: Frantar, E. et al. (2022) — same GPTQ paper as Phase 9 step 6, reused unchanged at `bits=8`.
- **Step 3 — NPU vs. CPU vs. GPU cost-comparison model** (`cost_model/`) [`19c5853`]: no dedicated citation.
- **Step 4 — Operator coverage analysis** (`op_coverage/`) [`39e7253`]: no dedicated citation.
- **Step 5 — Power/thermal modeling** (`thermal/`) [`fd93b8b`]: no dedicated citation — mirrors Phase 7 step 25's thermal-router split.
- **Step 6 — Cost model + placement pass integration** [`5174a69`]: no dedicated citation — edits Phase 4 steps 9 and 13.
- **Step 7 — ServingRouter integration** [`ab57541`]: no dedicated citation — edits Phase 9 step 8.

**Vendor docs:** Apple *Core ML* documentation (ANE targeting), ONNX
Runtime *NPU execution providers* documentation — both referenced as the
still-unrun `npu_onnx_export.py` path.

---

## Phase 16: Containerized & Orchestrated Deployment

**Start here:** `containers/DESIGN.md`, `containers/README.md`.

- **Step 1 — Dockerfile for the portable build** (`Dockerfile`) [`143dd60`]: no dedicated citation; see Docker docs under Vendor docs.
- **Step 2 — cgroup interaction with host-level tuning, measured**: **not yet implemented** (needs Docker, not installed locally — see project memory on the standing no-new-installs decision); no dedicated citation.
- **Step 3 — GPU passthrough config** (`docker/gpu/`) [`14a7706`]: no dedicated citation; see NVIDIA Container Toolkit docs under Vendor docs.
- **Step 4 — Kubernetes manifests for inference serving** (`k8s/serving/`) [`578bf18`, prerequisite: `9836f1e` "Add serving_daemon: real long-running process for k8s/serving/deployment.yaml"]: no dedicated citation; see Kubernetes docs (HPA custom-metrics API) under Vendor docs.
- **Step 5 — Gang-scheduled manifests for distributed training** (`k8s/training/`) [`4130c0d`, prerequisite: `5a386ff` "Add training_worker: real process-per-rank distributed training driver"]: no dedicated citation; see Kubernetes docs (`StatefulSet`/`podManagementPolicy`) under Vendor docs.
- **Step 6 — Device plugin gap analysis for FPGA/TPU** [`ecfa2e4`]: no dedicated citation.
- **Step 7 — Custom schedulers vs. Kubernetes, compared directly** [`adea9e6`]: no dedicated citation — cross-references Phase 5 steps 16 and 23 (`topo_scheduler`, `multitenancy`).

---

## Phase 17: Analog & Unconventional Compute Hardware — CODE COMPLETE (8/8 steps)

**Start here:** `analog_engine/README.md` and `analog_engine/DESIGN.md`
(phase-level wrap-up, `21a2ba7`), then each step's own README:
`device_model/`, `crossbar_mac/`, `nvm_comparison/`, `energy_model/`,
`dataflow_model/`, `systolic_sweep/`, `codesign_case_study/`,
`circuit_transient/`. Scoped in `e523a33`.

- **Step 1 — Device non-ideality model** (`analog_engine/device_model/`) [`734fb5d`]: Yu, S. (2018), *"Neuro-Inspired Computing with Emerging Nonvolatile Memory"* (Proceedings of the IEEE) — the device-physics review this step's noise/drift/endurance model draws on. Real bug caught and fixed by the test: the endurance check originally re-rolled a Bernoulli stuck-probability on every write, compounding into a wildly inflated failure rate; fixed with a per-cell fixed failure-threshold percentile compared against a cumulative failure curve.
- **Step 2 — Resistive crossbar MAC simulation** (`analog_engine/crossbar_mac/`) [`c96dc08`]: Shafiee, A. et al. (2016), *"ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars"* — the canonical resistive-crossbar-as-MAC-engine architecture this step simulates; Chi, P. et al. (2016), *"PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-Based Main Memory"* — a second foundational crossbar-PIM architecture, useful contrast; Ambrogio, S. et al. (2018), *"Equivalent-accuracy accelerated neural-network training using analog memory"* (Nature) — a real measured analog-training result, useful ground-truth calibration for the accuracy-vs-noise curve. Real finding: precision (num_levels) drives MAC accuracy (23.7%->1.8% relative RMSE, 4->64 levels) but crossbar SIZE doesn't (~5-7% relative RMSE flat from 8x8->64x64) — signal and noise both scale as sqrt(M) for random weights.
- **Step 3 — Non-volatile memory tradeoff comparison** (`analog_engine/nvm_comparison/`) [`7f16330`]: Yu (2018), same as step 1. Real finding: RRAM ranks first in an illustrative composite figure-of-merit despite not leading on any single axis (STT-MRAM has ~1000x its endurance, SRAM-CIM 1000x endurance + 20x lower write energy) — RRAM has no hard disqualifier, unlike SRAM-CIM (zero retention, volatile) or STT-MRAM (2 analog levels, structurally near-binary).
- **Step 4 — Energy model: analog MAC vs. digital MAC** (`analog_engine/energy_model/`) [`f2be889`]: Wu, Y.N., Emer, J. & Sze, V. (2019), *"Accelergy: An Architecture-Level Energy Estimation Methodology for Accelerator Designs"* — the energy-estimation methodology this step's shape mirrors. Real finding: ADC overhead dominates analog MAC energy by 121x-361x (pure-compute vs. realistic figure); at 32-level precision, realistic analog (1.505 pJ/MAC) is ~6x WORSE than a purpose-built NPU (0.253 pJ/MAC) though still beats GPU/CPU — a genuinely counter-intuitive result, echoing a real debate in the compute-in-memory literature.
- **Step 5 — PPA/dataflow modeling** (`analog_engine/dataflow_model/`) [`0529c3f`]: Sze, V., Chen, Y-H., Yang, T-J. & Emer, J. (2017), *"Efficient Processing of Deep Neural Networks: A Tutorial and Survey"* — the canonical reference for the Weight-/Output-/Input-/Row-Stationary taxonomy this step implements; Parashar, A. et al. (2019), *"Timeloop: A Systematic Approach to DNN Accelerator Evaluation"* — the tool this step reproduces a minimal version of; Wu, Emer & Sze (2019), same as step 4. Real finding: Row-Stationary numerically achieves WS's minimal weight movement AND IS's minimal input movement simultaneously; misaligned M=513 drops PE utilization to 0.8016, reproducing `tpu_engine/mxu_opt`'s independently-built utilization-cliff finding.
- **Step 6 — Systolic array design-space sweep** (`analog_engine/systolic_sweep/`) [`62a4f7e`]: Chen, Y-H., Emer, J. & Sze, V. (2016/2017), *"Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks"* — the concrete Row-Stationary accelerator design this step and step 5 are modeled after; Sze et al. (2017), same as step 5. Real finding: on 7 real GEMM shapes from `transformer/`'s actual trained config, a 128x128 array averages 3.79% utilization vs. 100% at each shape's best-matched size — independently reproduces `tpu_engine/mxu_opt`'s utilization-cliff finding from the opposite sweep direction.
- **Step 7 — Hardware-algorithm co-design case study** (`analog_engine/codesign_case_study/`) [`b64d802`]: no new citation — re-derives Phase 9 step 6's real `GptqQuantizer` (Frantar et al. 2022) under analog constraints via step 1's device model. Real finding: weight RMSE behaves exactly as physics predicts (noise always hurts, precision always helps); a real non-monotonic end-task perplexity result (analog noise very slightly IMPROVED perplexity on this overfit toy corpus) was caught by the test and led to replacing an overly strong assertion with a defensible one — analog noise's perplexity effect stays smaller than GPTQ's own quantization effect.
- **Step 8 — Analog circuit transient surrogate** (`analog_engine/circuit_transient/`) [`3c37b70`]: no dedicated citation — basic circuit theory (Ohm's law, Kirchhoff's current law) is the entire mathematical content needed; see the Background note below. Real finding: settling time scales quadratically with crossbar size via a distributed Elmore-style RC line (256x slower for a 16x size increase) — a third, independent reason bigger crossbars aren't free, on top of step 2's statistics-based finding.

**Background:** basic circuit theory (Ohm's law, Kirchhoff's current law)
is the entire mathematical content behind "a resistive crossbar computes a
matrix-vector product for free" (steps 2, 8) — no deeper circuit-design
text is needed; Sze et al. (2017) is also the right on-ramp for anyone who
hasn't seen the dataflow-taxonomy material before starting step 5.

---

## Phase 18: Dynamical Systems, SciML & Physics-Informed Architectures — CODE COMPLETE (10/10 steps)

**Start here:** `sciml/README.md` and `sciml/DESIGN.md` (phase-level
wrap-up, `372e9d3`), then each step's own README: `ode_solver/`,
`stiffness/`, `sde_solver/`, `neural_ode/`, `deq/`, `ssm_layer/`,
`diffusion/`, `ebm/`, `mup_scaling/`, `noise_aware_training/`. Scoped in
`e523a33`.

- **Step 1 — ODE solver library** (`sciml/ode_solver/`) [`7e475e8`]: no dedicated citation; see Iserles under Background below. Verified against closed-form solutions of three test ODEs (exponential decay, harmonic oscillator, logistic growth); RK4's empirical convergence order measured at 16.11x error reduction per `dt` halving (theory: 16x for a 4th-order method).
- **Step 2 — Stability & stiffness analysis** (`sciml/stiffness/`) [`62b27a9`]: no dedicated citation; same Iserles background as step 1. Forward Euler's exact stability boundary `dt < 2/|lambda|` measured directly (bounded just below, diverges to `1.1e12` just above); Van der Pol's stiffness ratio measured to grow from complex eigenvalues (non-stiff) at mu=1 to a ~2498 real-eigenvalue ratio at mu=50, via a numerically estimated (finite-difference) Jacobian.
- **Step 3 — SDE solver** (`sciml/sde_solver/`) [`68dbdd8`]: Kloeden, P.E. & Platen, E. (1992), *Numerical Solution of Stochastic Differential Equations* — the standard reference for Euler-Maruyama/Milstein and their convergence-order guarantees. Empirical strong-convergence order measured almost exactly on theory (Euler-Maruyama 1.38 vs. theoretical 1.41; Milstein 2.02 vs. theoretical 2.0, both on GBM's multiplicative-noise problem); Milstein verified to reduce to Euler-Maruyama exactly (`0.00e+00` difference) on Ornstein-Uhlenbeck's additive noise, a direct structural check.
- **Step 4 — Neural ODEs** (`sciml/neural_ode/`) [`36fd200`]: Chen, R.T.Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. (2018), *"Neural Ordinary Differential Equations"* (NeurIPS, best paper) — parameterize `dx/dt` with a network, backprop via the adjoint method instead of unrolling; Pontryagin, L.S. et al. (1962), *The Mathematical Theory of Optimal Control Processes* — the classical origin of the adjoint/costate method Chen et al. repurpose, useful for understanding *why* the adjoint ODE gives the right gradient. Real bug caught by the finite-difference gradient check: a `dtheta/dtau` sign flip from the reversed-time substitution (every parameter's relative error at exactly 2.0, i.e. `numeric=-analytic`), fixed to ~0 relative error; the reversed-time `x0` reconstruction (the adjoint method's O(1)-memory property) was already exact (`8.33e-17`) before the fix.
- **Step 5 — Deep Equilibrium Models** (`sciml/deq/`) [`3a38e89`]: Bai, S., Kolter, J.Z. & Koltun, V. (2019), *"Deep Equilibrium Models"* (NeurIPS) — the fixed-point-layer formulation and implicit-function-theorem backprop; Broyden, C.G. (1965), *"A Class of Methods for Solving Nonlinear Simultaneous Equations"* — the quasi-Newton fixed-point solver noted as a possible upgrade over the plain fixed-point iteration actually used. Gradient check matched finite differences on the first run (no sign bug, unlike step 4) — median relative error 0.000000, max 0.000003 across 21 parameters, each re-solving the fixed point from scratch.
- **Step 6 — State-space model layer** (`sciml/ssm_layer/`) [`a4d5843`]: Gu, A., Goel, K. & Ré, C. (2021), *"Efficiently Modeling Long Sequences with Structured State Spaces"* (S4) — the linear state-space recurrence layer this step implements (generic/non-HiPPO, a disclosed scope choice); Gu, A. & Dao, T. (2023), *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"* — the input-dependent (selective) extension, background only. Op-count scaling measured almost exactly on theory (SSM ratio 2.00, attention 3.83 vs. theoretical 4 on doubling L); attention beats the generic SSM on a long-range copy task (0.2321 vs. 0.3145 final loss) — the literature-motivated reason S4's real contribution is HiPPO initialization, not just a linear recurrence.
- **Step 7 — Diffusion / flow-matching generative model** (`sciml/diffusion/`) [`9cca103`]: Sohl-Dickstein, J. et al. (2015), *"Deep Unsupervised Learning using Nonequilibrium Thermodynamics"* — the original diffusion-model formulation; Ho, J., Jain, A. & Abbeel, P. (2020), *"Denoising Diffusion Probabilistic Models"* (DDPM) — the practical forward-noising/learned-reverse-process formulation this step implements. DDPM only, not flow matching (Lipman et al. 2023) — a disclosed scope reduction. Real result: 200 generated samples split 94/106 across a toy two-cluster target (close to true 50/50, not mode-collapsed), each cluster's empirical mean within 0.58/0.32 of the true center.
- **Step 8 — Energy-based model** (`sciml/ebm/`) [`904fe1a`]: LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. & Huang, F. (2006), *"A Tutorial on Energy-Based Learning"* — the EBM formulation; Hinton, G. (2002), *"Training Products of Experts by Minimizing Contrastive Divergence"* — the training procedure this step implements, with short-run Langevin MCMC negative sampling. Real, literature-consistent finding from a direct same-binary comparison to step 7: diffusion (1.04 mean distance to true cluster centers) clearly beats the EBM (2.36) at deliberately comparable budgets — the well-documented real difficulty of short-chain CD training vs. diffusion's tractable objective, not a bug.
- **Step 9 — muP-style scaling study** (`sciml/mup_scaling/`) [`6c0f30f`]: Yang, G. & Hu, E.J. et al. (2021/2022), *"Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"* (muP) — the entire claim under test: hyperparameters tuned at small width transfer to large width under the right parameterization. Tests the output-layer-LR-scaling mechanism specifically (disclosed scope reduction from the full multi-parameter table). Real result: muP's best LR was exactly 1.0 at all three widths tested (4/8/16); SP's best LR shifted 0.3->0.3->0.1 — an exact confirmation with no tuning to force it, plus SP diverging at 7/15 (width,LR) points vs. muP's 2/15.
- **Step 10 — Physics-informed / noise-aware training bridge** (`sciml/noise_aware_training/`) [`6c3d29b`]: no new citation — trains a real model (step 9's MLP) through Phase 17 step 1's device-noise model, structurally mirroring Phase 14 step 5's adversarial-training loop with the adversary replaced by device noise. Real bug caught by catastrophic training divergence: naive finite-differencing through the quantization+noise pipeline gave a degenerate zero-or-spike gradient — exactly why real QAT needs a Straight-Through Estimator; fixed, then found the robustness benefit needed a stronger noise regime to become measurable, ending in a clean echo of Phase 14's robustness/accuracy tradeoff (noisy loss cut ~45% at a real clean-accuracy cost).

**Background:** an ODE/PDE course covering explicit-vs-implicit solvers and
stability (Iserles, *A First Course in the Numerical Analysis of
Differential Equations*) is the right on-ramp before step 1; Goodfellow,
Bengio & Courville's *Deep Learning*, ch. 20 (generative models) is useful
connective tissue between steps 7 and 8 if the primary sources above move
too fast.

---

## Phase 19: Framework-Native Training (PyTorch/JAX) — scoped, not yet built

**Start here:** none yet — `framework_native/DESIGN.md` will exist once
step 1 lands. Scoped in `e523a33`; no implementation commits yet.

- **Step 1 — PyTorch port of `transformer/`**: **not yet implemented**; Paszke, A. et al. (2019), *"PyTorch: An Imperative Style, High-Performance Deep Learning Library"* — the autograd/eager-execution design this step targets directly.
- **Step 2 — `torch.compile` benchmarking**: **not yet implemented**; no academic citation planned; see PyTorch's TorchDynamo/TorchInductor documentation under Vendor docs — needed to interpret the step's speedup-or-not result rather than treat it as a black box.
- **Step 3 — DDP over CPU (`gloo`)**: **not yet implemented**; Li, S. et al. (2020), *"PyTorch Distributed: Experiences on Accelerating Data Parallel Training"* — the real DDP paper (gradient bucketing, overlapping backward with all-reduce) this step will be built on and compared against Phase 6 step 3's hand-written `data_parallel`.
- **Step 4 — FSDP vs. hand-written ZeRO**: **not yet implemented**; Zhao, Y. et al. (2023), *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* — the direct paper-level counterpart to compare against Phase 6 steps 7-9's `zero1`/`zero2`/`zero3` (FSDP is essentially ZeRO stage 3 as a first-class PyTorch API — see Rajbhandari et al. 2020 under Phase 6 step 7).
- **Step 5 — JAX port**: **not yet implemented**; Bradbury, J. et al. (2018), *JAX: composable transformations of Python+NumPy programs* — the `jit`/`grad`/`vmap`/`pmap` functional-transform model this step targets; already partially in use for Phase 8's `tpu_engine`.
- **Step 6 — One real run through a production framework**: **not yet implemented**; Rasley, J. et al. (2020), *"DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters"* — if DeepSpeed ends up used; Moritz, P. et al. (2018), *"Ray: A Distributed Framework for Emerging AI Applications"* — if Ray Train ends up used instead. Documentation for whichever framework is actually chosen once step 6 lands.

**Background:** none beyond what Phase 6 already assumes (backprop,
optimizers, data/tensor/pipeline parallelism) — this phase is explicitly
about mapping already-understood concepts onto specific framework APIs,
not new theory.

---

## Cross-cutting: testing & methodology

These apply across every phase, not to one step:

- **Commit-per-step convention** — this repo commits (and pushes) almost
  exactly once per PLAN.md step, with "Phase N step M" (or, for Phase 1,
  "(Phase 1, step M)") in the commit message — see project memory:
  `feedback_commit_push`. That convention is what makes the `[hash]`
  annotations above possible: every step above with a hash was found via
  `git log --oneline | grep "Phase N step M"`, not guessed. Three phases
  (3's steps 6-24, 4's steps 1-15, 5's steps 1-25) landed as one bulk
  commit each instead of per-step, from an earlier "stub everything, then
  fill in" pass before the per-step convention solidified — noted inline
  where that applies.
- **Property-based testing** — Claessen & Hughes (2000), *QuickCheck*
  (Phase 1 step 17); reused directly by Phase 9 step 1's 300-trial
  `paged_kv` property test.
- **TSan/ASan/UBSan discipline** — every concurrent structure (Phase 1
  steps 3-13) must be race-clean before a step counts as done (see project
  memory: `feedback_tsan_discipline`). Background: Serebryany, K. &
  Iskhodzhanov, T. (2009), *"ThreadSanitizer: Data Race Detection in
  Practice"*.
- **Formal verification** — Phase 7 step 21 and Phase 10 step 5's
  k-induction model checking; Sheeran, M., Singh, S. & Stålmarck, G.
  (2000), *"Checking Safety Properties Using Induction and a SAT-Solver"*
  is the standard reference for the k-induction technique SymbiYosys's
  `mode prove` uses.
- **Roofline modeling** — Williams et al. (2009), used verbatim in Phase 2
  step 10 and Phase 3 step 21, and structurally in Phase 8's MXU/HBM
  analysis (steps 5-6, 9).

---

## Full bibliography (alphabetical, by first author)

Arthur & Vassilvitskii (2007) k-means++ · Bentley (1975) multidimensional
binary search trees · Bergstra & Bengio (2012) random search · Bergstra et
al. (2011) TPE · Bischl et al. (2019) OpenML-CC18 · Bradley & Terry (1952)
pairwise comparison model · Breiman (2001) random forests · Breiman et al.
(1984) CART · Chandy & Lamport (1985) distributed snapshots · Chase & Lev
(2005) work-stealing deque · Christiano et al. (2017) deep RL from human
preferences · Claessen & Hughes (2000) QuickCheck · Clopper & Pearson (1934)
confidence limits · Cohen, Rosenfeld & Kolter (2019) randomized smoothing ·
Dao et al. (2022) FlashAttention · Dao et al. (2023) Flash-Decoding ·
Desnoyers et al. (2012) URCU · Fedus, Zoph & Shazeer (2021) Switch
Transformers · Fidge/Mattern (1988) vector clocks · Frantar et al. (2022)
GPTQ · Friedman (2001) gradient boosting machine · Friedman (2002)
stochastic gradient boosting · Goodfellow, Shlens & Szegedy (2014) FGSM ·
Gregg, *BPF Performance Tools* · Halko, Martinsson & Tropp (2011) randomized
SVD · Herlihy & Shavit, *The Art of Multiprocessor Programming* · Huang et
al. (2019) GPipe · Ioffe & Szegedy (2015) batch normalization · Jaderberg et
al. (2017) Population Based Training · Jamieson & Talwalkar (2016)
successive halving · Jouppi et al. (2017/2023) TPU papers · Kwon et al.
(2023) PagedAttention/vLLM · Lamport (1978) time/clocks/ordering · Lamport
(2002) *Specifying Systems* · Lattner et al. (2021) MLIR · Lattner & Adve
(2004) LLVM · Leviathan, Kalman & Matias (2023) speculative decoding · Lewis
et al. (2020) RAG · Li et al. (2016) Hyperband · Li et al. (2018) ASHA ·
Liu et al. (2004) approximate ball-tree ANN · Madry et al. (2017) PGD /
adversarial training · Malkov & Yashunin (2016/2018) HNSW · Michael (2004)
hazard pointers · Michael & Scott (1996) lock-free queues · Micikevicius et
al. (2017) mixed precision training · Milakov & Gimelshein (2018) online
softmax · Mockus (1978) Expected Improvement · Narayanan et al. (PipeDream /
Megatron-LM pipelining) · Nocedal & Wright, *Numerical Optimization* ·
Ongaro & Ousterhout (2014) Raft · Ongaro (2014) Raft thesis · Ouyang et al.
(2022) InstructGPT/RLHF · Papernot et al. (2016) transferability · Platt
(1998) SMO · Rabenseifner (2004) collective reduction optimization ·
Radford et al. (2021) CLIP · Rafailov et al. (2023) DPO · Rajbhandari et al.
(2020) ZeRO · Rajbhandari et al. (2021) ZeRO-Infinity · Rasmussen & Williams
(2006) Gaussian Processes for ML · Schulman et al. (2017) PPO · Serebryany &
Iskhodzhanov (2009) ThreadSanitizer · Shazeer et al. (2017) sparsely-gated
MoE · Sheeran, Singh & Stålmarck (2000) k-induction · Shoeybi et al. (2019)
Megatron-LM · Sigelman et al. (2010) Dapper · Stiennon et al. (2020) learning
to summarize from human feedback · Sutton & Barto, *Reinforcement Learning:
An Introduction* · Szegedy et al. (2013) intriguing properties of neural
networks · Tsipras et al. (2018) robustness-accuracy tradeoff · Uhlmann
(1991) ball trees · van den Oord, Li & Vinyals (2018) InfoNCE · Vaswani et
al. (2017) Attention Is All You Need · Williams, Waterman & Patterson (2009)
roofline model · Zou & Hastie (2005) elastic net.

**Phases 17-19 additions:** Ambrogio et al. (2018) analog in-memory training
· Bai, Kolter & Koltun (2019) Deep Equilibrium Models · Bradbury et al.
(2018) JAX · Broyden (1965) quasi-Newton nonlinear solver · Chen, Emer &
Sze (2016/2017) Eyeriss · Chen, Rubanova, Bettencourt & Duvenaud (2018)
Neural ODEs · Chi et al. (2016) PRIME · Gu, Goel & Ré (2021) S4 · Gu & Dao
(2023) Mamba · Hinton (2002) contrastive divergence · Ho, Jain & Abbeel
(2020) DDPM · Iserles, *A First Course in the Numerical Analysis of
Differential Equations* · Kloeden & Platen (1992) numerical SDE methods ·
LeCun, Chopra, Hadsell, Ranzato & Huang (2006) energy-based learning
tutorial · Li et al. (2020) PyTorch DDP · Lipman et al. (2023) Flow
Matching · Moritz et al. (2018) Ray · Parashar et al. (2019) Timeloop ·
Paszke et al. (2019) PyTorch · Pontryagin et al. (1962) optimal control
(adjoint method origin) · Rasley et al. (2020) DeepSpeed · Shafiee et al.
(2016) ISAAC · Sohl-Dickstein et al. (2015) diffusion origin · Sze, Chen,
Yang & Emer (2017) DNN accelerator survey (WS/OS/IS/RS taxonomy) · Wu,
Emer & Sze (2019) Accelergy · Yang & Hu et al. (2021/2022) muP / Tensor
Programs V · Yu, S. (2018) nonvolatile-memory neuro-inspired computing
review · Zhao et al. (2023) PyTorch FSDP.

---

## Vendor docs / specs / manuals index

| Area | Doc |
|---|---|
| CPU | Intel 64/IA-32 Optimization Reference Manual |
| GPU | CUDA C++ Programming Guide, PTX ISA, NVML API reference, Nsight Compute/Systems docs, Hopper Architecture Whitepaper, Triton language docs, CUTLASS docs |
| Compiler | MLIR language reference, TableGen ODS reference, LLVM IR reference manual, libFuzzer docs |
| Networking | libfabric/EFA programmer's guide, gRPC C++ docs, FlatBuffers docs, AF_XDP kernel docs, NCCL developer guide, IEEE 1588 (PTP) |
| Distributed systems | TLA+ (*Specifying Systems*), TLC model checker docs |
| FPGA | Vivado Design Suite UG901/UG904, Vitis HLS UG1399, UltraScale+ architecture docs, XRT architecture/API docs, Vitis AI User Guide, P4_16 Language Specification, cocotb docs, SymbiYosys/Yosys docs |
| TPU | XLA docs, StableHLO specification, JAX docs (`pjit`, `jax.export`) |
| NPU | Apple Core ML docs, ONNX Runtime NPU execution providers docs |
| Observability | OpenTelemetry OTLP specification, Anthropic API reference (Messages API, structured outputs) |
| Containers | Docker docs, NVIDIA Container Toolkit docs, Kubernetes docs (HPA custom metrics, StatefulSet) |
| Analog/unconventional compute | No vendor toolchain exists to document (no analog/neuromorphic silicon or SDK) — Phase 17's "vendor docs" are the primary-source papers above (ISAAC, PRIME, Timeloop, Accelergy) |
| SciML/dynamical systems | `scipy.integrate` docs (reference ODE/SDE solver implementations to check against), `diffrax`/`torchdiffeq` docs (existing Neural-ODE libraries, useful for API-shape comparison even if not depended on) |
| Framework-native training | PyTorch docs (`torch.autograd`, `torch.compile`, `torch.distributed`, FSDP), JAX docs (`jit`/`vmap`/`pmap`/`grad`, `jax.sharding`), PyTorch Lightning docs, DeepSpeed docs, Ray Train docs |

---

## Not covered here

**Phase 11 (Polish + Portfolio)** — PLAN.md defines this phase but it isn't
implementation work with its own citations; it's write-up/presentation of
everything above, so it has no separate reading list entry.
