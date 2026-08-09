# Cross-Machine Runtime — Reading List

Everything needed to understand this repo as deeply as the assistant working on
it does: the in-repo docs to read (and in what order), the external papers
those docs cite or build on, and the vendor/spec manuals the hardware-gated
code targets. PLAN.md and SCOPE.md describe *what to build*; this document is
the *background + citation* layer underneath them, assembled by reading every
README.md/DESIGN.md and grepping every source comment in the repo for actual
citations, then filling gaps with the general background those citations
assume.

## How to use this

- Read in the order below — it's PLAN.md's phase order, which is also the
  dependency order (Phase 6 reuses Phase 5's collectives, Phase 9 reuses
  Phase 6's transformer, Phase 13 reuses Phase 6+9, etc.).
- For each phase: read the "Start here" in-repo docs first, skim the step
  table to know what exists, then read external sources only for the steps
  you actually want to go deep on — you don't need every paper to understand
  the repo, only the ones behind steps you're studying closely.
- **The in-repo docs are the primary source.** Every step's own README.md
  documents what was actually built and what was actually measured (or, for
  hardware-gated steps, what's still TODO) — the papers below are the
  background those READMEs assume, not a replacement for reading them.
- The final two sections are flat indexes (alphabetical bibliography; vendor
  docs) for looking a specific citation up without walking the whole phase
  structure.
- **Phases 17-19** (added 2026-08-09, scoped in PLAN.md/SCOPE.md/CLAUDE.md,
  not yet implemented) close a gap found by comparing this repo against a
  batch of analog/physics-based-AI-compute job descriptions: analog/
  unconventional compute hardware, dynamical-systems/SciML and the
  "unconventional" model architectures those roles build, and genuine
  PyTorch/JAX framework fluency. Their sections below have no in-repo doc
  pointers yet (there's no code to point to) — they're pure background
  reading to have in hand before that code gets written, and will gain
  "Start here" pointers once each step exists.

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

**Start here:** `foundation/DESIGN.md`, `foundation/README.md` (if present),
each subdirectory's own README.md.

| Directory | Topic |
|---|---|
| `foundation/chase_lev/` | Work-stealing deque |
| `foundation/mpmc_queue.h`, `foundation/freelist/` | Lock-free MPMC queue, freelist allocator |
| `foundation/hazard/` | Hazard pointers |
| `foundation/epoch/` | Epoch-based reclamation |
| `foundation/rcu/` | Userspace RCU |
| `foundation/aba/` | ABA problem demonstration |
| `foundation/arena/` | Arena allocator |
| `foundation/coro/` | C++20 coroutine engine |
| `foundation/ws_pool/` | Work-stealing thread pool |
| `foundation/proptest*` | Property-based testing framework |
| `foundation/perf/` | Hardware performance-counter infrastructure |

**Primary sources:**
- Chase, D. & Lev, Y. (2005), *"Dynamic Circular Work-Stealing Deque"* — the
  algorithm `foundation/chase_lev/chase_lev.h` implements directly.
- Michael, M. & Scott, M. (1996), *"Simple, Fast, and Practical Non-Blocking
  and Blocking Concurrent Queue Algorithms"` — the lineage `mpmc_queue.h`
  descends from.
- Desnoyers, M. et al. (2012), *"User-Level Implementations of Read-Copy
  Update"* — `foundation/rcu/rcu_domain.h` cites this directly ("based on
  URCU, Desnoyers et al. 2012") for the per-thread-counter algorithm.
- Michael, M. (2004), *"Hazard Pointers: Safe Memory Reclamation for
  Lock-Free Objects"* — background for `foundation/hazard/`.
- Baker, L., cppcoro (CppCon 2019 talk) — `foundation/coro/coro.h` cites this
  directly for the `AsyncMutex` design.
- Claessen, K. & Hughes, J. (2000), *"QuickCheck: A Lightweight Tool for
  Random Testing of Haskell Programs"* — the property-based-testing paradigm
  `foundation/proptest` implements in C++.

**Background:** Herlihy, M. & Shavit, N., *The Art of Multiprocessor
Programming* — the standard textbook covering ABA, hazard pointers, and
work-stealing in one place; useful as connective tissue between the papers
above.

---

## Phase 2: CPU Backend — affinity, SIMD, tiling, inference engine

**Start here:** `cpu_engine/DESIGN.md`, per-step READMEs (`affinity/`,
`hugepage/`, `os_tuning/`, `avx512/`, `branchless/`, `nt_store/`,
`prefetch/`, `tiling/`, `inference/`, `roofline/`, `perf_deep_dive/`,
`pgo/`, `busy_poll/`).

**Primary sources:**
- Williams, S., Waterman, A. & Patterson, D. (2009), *"Roofline: An
  Insightful Visual Performance Model for Multicore Architectures"* —
  `cpu_engine/roofline/roofline.h` and `gpu_engine/roofline/roofline.h` both
  cite this directly; the roofline model is the standard "is this kernel
  compute-bound or memory-bound" lens used across CPU, GPU, and TPU phases.
- Intel 64 and IA-32 Architectures Optimization Reference Manual — AVX-512
  instruction latencies/throughput, used for `avx512/`.
- Drepper, U. (2007), *"What Every Programmer Should Know About Memory"* —
  hugepages, NUMA, prefetching background for `hugepage/`, `prefetch/`.

**Background:** PGO (`pgo/`) assumes familiarity with profile-guided
compilation (`-fprofile-generate`/`-fprofile-use`); busy-polling (`busy_poll/`)
assumes familiarity with the latency/CPU-utilization tradeoff vs. blocking
syscalls (`epoll`/`futex`).

---

## Phase 3: GPU Backend — CUDA (code-complete, hardware-gated)

**Start here:** `gpu_engine/DESIGN.md`, per-step READMEs. None of this phase
has run on real hardware (no CUDA toolchain on Mac) — every README's
`## Results` is `TODO: run on [hardware]`.

| Directory | Topic |
|---|---|
| `warp_primitives/`, `coalescing/`, `occupancy/` | Warp shuffle/vote, memory coalescing, occupancy tuning |
| `kernels/` | Elementwise + GEMM kernels |
| `ptx_sass/` | PTX/SASS inspection |
| `flash_attn/` | Flash attention |
| `graphs/` | CUDA graphs |
| `p2p/` | Peer-to-peer transfers |
| `precision/` | Mixed precision, FP8, tensor-core alignment |
| `hopper/` | Hopper TMA/WGMMA |
| `sparsity/` | 2:4 structured sparsity |
| `roofline/`, `mps/`, `power/`, `nsight_ci/` | Roofline, MPS, NVML power monitoring, Nsight CI |

**Primary sources:**
- Dao, T., Fu, D., Ermon, S., Rudra, A. & Ré, C. (2022), *"FlashAttention:
  Fast and Memory-Efficient Exact Attention with IO-Awareness"* —
  `gpu_engine/flash_attn/README.md` cites this directly.
- Milakov, M. & Gimelshein, N. (2018), *"Online normalizer calculation for
  softmax"* — the running-max/normalizer online-softmax trick
  `flash_attn/README.md` also cites, used inside the flash-attention kernel.
- Micikevicius, P. et al. (2017), *"Mixed Precision Training"* — background
  for `precision/mixed_precision.h`.
- NVIDIA, *CUDA C++ Programming Guide* — the primary reference for streams,
  warp primitives, occupancy, and graphs.
- NVIDIA, *Parallel Thread Execution (PTX) ISA* — needed to read
  `ptx_sass/`'s output.
- NVIDIA, *Hopper Architecture Whitepaper* — TMA/WGMMA background for
  `hopper/`.

**Vendor docs:** CUDA Programming Guide, PTX ISA reference, `nvcc` docs,
Nsight Compute/Systems docs, NVML API reference — all needed once this phase
is validated on real hardware.

---

## Phase 4: Compiler / IR (MLIR) — code-complete, hardware-gated

**Start here:** `compiler/DESIGN.md`, `compiler/README.md`. Only
`compiler/cost_model/` has actually run locally (no MLIR dependency, plain
`clang++`); everything else needs an LLVM/MLIR toolchain build, deferred to
Linux hardware validation.

| Directory | Topic |
|---|---|
| `dialect/` | `runtime` dialect: 15 ops, 3 attrs, TableGen |
| `shape_inference/`, `fusion/`, `affine_lower/` | Shape inference, op fusion, affine lowering/tiling |
| `mem_planning/`, `remat/` | Memory planning, rematerialization |
| `placement/`, `sharding/` | Device placement, auto-sharding |
| `kernel_spec/` | Kernel specialization |
| `aot/` | AOT pipeline: orchestrates all passes + LLVM codegen + link |
| `cost_model/` | Device cost model (the one piece actually run locally) |
| `fuzzing/`, `upstream/`, `mlir_setup/` | Compiler fuzzing, upstream MLIR notes, toolchain setup |

**Primary sources:**
- Lattner, C. et al. (2021), *"MLIR: Scaling Compiler Infrastructure for
  Domain Specific Computation"* (CGO 2021) — the foundational paper for the
  whole phase's IR design (TableGen ops, dialect conversion, pass
  pipelines).
- Chris Lattner & Vikram Adve, *"LLVM: A Compilation Framework for Lifelong
  Program Analysis & Transformation"* (CGO 2004) — LLVM codegen background
  for the AOT pipeline's backend.

**Background:** rematerialization (`remat/`) draws on the classic
register-allocation-by-graph-coloring / spill-cost literature (Chaitin et
al.); auto-sharding (`sharding/`) parallels GSPMD/Megatron-style sharding
annotations (see Phase 6's Megatron-LM references below — same idea applied
at the compiler-IR level instead of the model-code level).

**Vendor docs:** MLIR language reference, TableGen ODS reference, LLVM IR
reference manual.

---

## Phase 5: Distributed Layer + Networking

**Start here:** `networking/DESIGN.md`, `networking/README.md` (has the full
per-step status table: 12 of 25 steps actually built+run on this Mac).

| Directory | Topic | Locally run? |
|---|---|---|
| `common/` | Portable `Channel` transport (real POSIX sockets) | yes |
| `rdma_v1/`, `efa_srd/` | RDMA/EFA (TCP baseline runs; real EFA gated) | partial |
| `ptp/` | PTP clock sync | gated |
| `grpc_control/`, `flatbuffers_data/` | gRPC control plane, FlatBuffers data plane | gated |
| `af_xdp/`, `userspace_net/` | AF_XDP / userspace networking | gated (Linux) |
| `nic_deep_dive/` | NIC architecture writeup | gated |
| `ring_allreduce/`, `halving_doubling/`, `tree_allreduce/` | All-reduce algorithms | yes |
| `collectives/` | Broadcast/reduce-scatter/all-gather library | yes |
| `nccl_tuning/` | NCCL tuning config | gated |
| `topo_scheduler/` | Topology-aware scheduler | yes |
| `vector_clocks/` | Lamport clocks + vector clocks | yes |
| `chandy_lamport/` | Distributed snapshots | yes |
| `raft/` | Raft consensus (leader election + log replication) | yes |
| `tla_raft/`, `tla_collective/` | TLA+ specs for Raft and the collective protocol | gated (needs Java/TLC) |
| `backpressure/`, `hedged_requests/`, `multitenancy/` | Backpressure, hedged requests, multi-tenancy | yes |
| `chaos/` | Chaos engineering harness | partial |

**Primary sources:**
- Lamport, L. (1978), *"Time, Clocks, and the Ordering of Events in a
  Distributed System"* — Lamport clocks, `networking/vector_clocks/`.
- Fidge/Mattern (independently, 1988) — vector clocks proper (the extension
  beyond Lamport's scalar clock that `VectorClock` implements).
- Chandy, K.M. & Lamport, L. (1985), *"Distributed Snapshots: Determining
  Global States of Distributed Systems"* — `networking/chandy_lamport/`
  cites this directly ("the classic algorithm, Chandy & Lamport, 1985").
- Ongaro, D. & Ousterhout, J. (2014), *"In Search of an Understandable
  Consensus Algorithm"* (Raft paper) — `networking/raft/` implements this;
  election timeout range (150–300ms) is taken directly from the paper.
- Ongaro, D. (2014), *Consensus: Bridging Theory and Practice* (Stanford PhD
  thesis) — `networking/tla_raft/README.md` cites the thesis appendix
  directly as the source the community TLA+ Raft spec (and this repo's
  scoped-down version) follows.
- Lamport, L. (2002), *Specifying Systems* — the TLA+ book; needed to read
  `tla_raft/Raft.tla` and `tla_collective/`.
- Rabenseifner, R. (2004), *"Optimization of Collective Reduction
  Operations"* — `networking/halving_doubling/README.md` names this
  directly ("Rabenseifner's algorithm") for the recursive halving-doubling
  all-reduce.
- Patarasuk, P. & Yuan, X. (2009), *"Bandwidth Optimal All-reduce Algorithms
  for Clusters of Workstations"* — background for the ring all-reduce's
  bandwidth-optimality claim in `ring_allreduce/`.
- IEEE 1588 (Precision Time Protocol) standard — `networking/ptp/`.

**Vendor docs:** libfabric/EFA programmer's guide, gRPC C++ docs,
FlatBuffers schema/codegen docs, AF_XDP kernel documentation, NCCL
developer guide.

---

## Phase 6: Distributed GPU Training (+ `/transformer/`)

**Start here:** `distributed_training/README.md`, plus `transformer/README.md`
for the model every RLHF-stage step (22–25) trains.

| Directory | Topic |
|---|---|
| `data_loading/`, `data_parallel/`, `grad_accum/`, `grad_clipping/` | Data pipeline, data parallelism, grad accumulation/clipping |
| `autograd/` | Reverse-mode autograd engine + toy MLP (also what Phase 14 attacks) |
| `zero1/`, `zero2/`, `zero3/`, `zero_infinity/` | ZeRO stages 1–3 + CPU/NVMe offload |
| `col_row_linear/` | Column/row-parallel linear (Megatron-LM tensor parallelism) |
| `tensor_parallel_attn/`, `seq_parallel/` | Tensor-parallel attention, sequence parallelism |
| `pipeline_1f1b/` | GPipe vs. 1F1B pipeline scheduling |
| `parallel_3d/` | 3D parallelism (data + tensor + pipeline) |
| `moe/` | Mixture-of-Experts / expert parallelism |
| `checkpoint/` | Sharded checkpointing |
| `compute_comm_overlap/` | Compute/communication overlap |
| `sync_batchnorm/` | SyncBatchNorm |
| `full_training_loop/` | End-to-end training loop |
| `sparsity_training/` | 2:4 structured sparsity training |
| `sft/` | Supervised fine-tuning |
| `reward_model/` | Reward model (Bradley-Terry) |
| `ppo_rlhf/` | PPO-based RLHF |
| `dpo/` | Direct Preference Optimization |
| `training_worker/` | Real process-per-rank training driver (added for Phase 16) |
| `gpudirect_storage/` | GPUDirect Storage (hardware-gated) |

**Primary sources:**
- Vaswani, A. et al. (2017), *"Attention Is All You Need"* — the transformer
  architecture `transformer/transformer_model.h` and
  `tensor_parallel_attn/attention.h` implement.
- Shoeybi, M. et al. (2019), *"Megatron-LM: Training Multi-Billion Parameter
  Language Models Using Model Parallelism"* — `col_row_linear/` and
  `tensor_parallel_attn/README.md` both cite Megatron-style tensor
  parallelism directly as the pattern being validated.
- Rajbhandari, S. et al. (2020), *"ZeRO: Memory Optimizations Toward
  Training Trillion Parameter Models"* — `zero1/`/`zero2/`/`zero3/`.
- Rajbhandari, S. et al. (2021), *"ZeRO-Infinity: Breaking the GPU Memory
  Wall for Extreme Scale Deep Learning"* — `zero_infinity/`.
- Huang, Y. et al. (2019), *"GPipe: Efficient Training of Giant Neural
  Networks using Pipeline Parallelism"* — the GPipe baseline
  `pipeline_1f1b/pipeline_schedule.h` compares 1F1B against directly.
- Narayanan, D. et al. (2019 PipeDream / 2021 Megatron-LM pipelining paper)
  — 1F1B scheduling; `pipeline_1f1b/pipeline_schedule.h` cites "see the
  Megatron-LM paper" for the bubble-time equivalence result it validates.
- Korthikanti, V. et al. (2022), *"Reducing Activation Recomputation in
  Large Transformer Models"* — sequence parallelism background for
  `seq_parallel/`.
- Shazeer, N. et al. (2017), *"Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer"* — `moe/`.
- Fedus, W., Zoph, B. & Shazeer, N. (2021), *"Switch Transformers"* —
  further MoE/expert-parallelism background for `moe/`.
- Ioffe, S. (2017), *"Batch Renormalization"* / Ioffe & Szegedy (2015)
  *"Batch Normalization"* — background for `sync_batchnorm/`'s
  cross-device statistics synchronization.
- Bradley, R.A. & Terry, M.E. (1952), *"Rank Analysis of Incomplete Block
  Designs"* — the Bradley-Terry pairwise-preference model
  `reward_model/reward_model.h` and `dpo/dpo.h` both build on directly.
- Ouyang, L. et al. (2022), *"Training language models to follow
  instructions with human feedback"* (InstructGPT/RLHF) — the
  SFT→reward-model→PPO pipeline `sft/`→`reward_model/`→`ppo_rlhf/` mirrors.
- Schulman, J. et al. (2017), *"Proximal Policy Optimization Algorithms"* —
  `ppo_rlhf/ppo_rlhf.h`'s clipped surrogate objective and "k1" KL estimator
  (cited directly as "Schulman's k1 estimator").
- Christiano, P. et al. (2017), *"Deep Reinforcement Learning from Human
  Preferences"* — the RLHF-from-preferences framing PPO/DPO both sit inside.
- Stiennon, N. et al. (2020), *"Learning to Summarize from Human Feedback"*
  — an early large-scale RLHF pipeline with the same reward-model+PPO shape.
- Rafailov, R. et al. (2023), *"Direct Preference Optimization: Your
  Language Model is Secretly a Reward Model"* — `dpo/dpo.h` and
  `dpo/README.md` cite this directly; the whole step is built on their
  closed-form-optimal-policy reparameterization.

**Background:** standard backprop/optimizer material (SGD, Adam) underlies
`autograd/`; policy-gradient/REINFORCE (Sutton & Barto, *Reinforcement
Learning: An Introduction*, ch. 13) is useful background before Schulman's
PPO paper.

---

## Phase 7: FPGA Backend — code-complete, hardware-gated

**Start here:** `fpga_engine/README.md` (per-step status table).

| Directory | Topic |
|---|---|
| `f1_setup/`, `tcl_pipeline/`, `power_ci/` | AWS F1 validation, TCL synth/impl/bitstream, Vivado power CI |
| `axi_stream/` | AXI4-Stream passthrough |
| `dot_product/`, `loop_opt/` | Dot-product II study, UNROLL/PIPELINE/DATAFLOW |
| `dsp_lut/`, `fixed_point/` | DSP48E2 vs LUT, `ap_fixed` precision/resource/latency |
| `bram_uram/`, `ddr4/` | BRAM vs URAM access patterns, multi-bank DDR4 |
| `dma/`, `pcie_latency/`, `pingpong/` | Host DMA via XRT, PCIe latency decomposition, double-buffered overlap |
| `ml_kernel/` | Fully pipelined INT8 MLP inference kernel |
| `timing_closure/`, `slr/`, `clock_gating/` | Critical-path/retiming, SLR partitioning, clock-gating power model |
| `xadc/` | Die temperature/voltage monitoring (real XRT sensor API) |
| `ila_debug/` | ILA debug core + AXI handshake checker |
| `cocotb/` | Verilog testbenches, **actually run** (Icarus Verilog), caught a real DMA timing bug |
| `symbiyosys/` | Formal verification (**actually run**, prebuilt OSS CAD Suite `yosys`/`sby`) |
| `partial_reconfig/` | Dynamic Function eXchange (DFX) hot-swap |
| `fpga_net/` | RDMA-like FPGA-direct network path (P4_16) |
| `vitis_ai/` | DPU-vs-custom-kernel comparison |
| `thermal_router/` | Thermal-aware routing |

**Primary sources:**
- Xilinx/AMD, *Vivado Design Suite User Guide* (UG901/UG904 etc.) — TCL
  synthesis/implementation flow.
- Xilinx/AMD, *Vitis HLS User Guide* (UG1399) — pragmas (`PIPELINE`,
  `UNROLL`, `DATAFLOW`), II, `ap_fixed`.
- Xilinx/AMD, *UltraScale+ Architecture* documents (DSP48E2 slice, SLR
  crossing, clocking).
- Xilinx/AMD, *XRT (Xilinx Runtime) Architecture and API* — the real DMA/
  sensor/`load_xclbin()` APIs `dma/`, `xadc/`, `partial_reconfig/` target.
- P4 Language Consortium, *P4_16 Language Specification* — `fpga_net/
  rdma_bypass.p4`.
- SymbiYosys / Yosys documentation (YosysHQ) — formal flow used in
  `symbiyosys/`; the README documents a real gap (free `yosys` lacks full
  SVA `assert property` grammar — needs the commercial Verific plugin) that
  forced rewriting properties into procedural `assert`/`$past`/`$stable`
  form.
- cocotb documentation — the Python-based testbench framework used in
  `cocotb/`.
- Xilinx/AMD, *Vitis AI User Guide* — DPU architecture, quantization flow
  (`vai_q_pytorch`), compilation (`vai_c_xir`), used in `vitis_ai/`.

**Background:** the DSP48E2-vs-LUT tradeoff (`dsp_lut/`) and timing-closure
retiming (`timing_closure/`) assume basic digital-design vocabulary
(pipeline registers, critical path, setup/hold) — any intro RTL/FPGA design
text (e.g. Harris & Harris, *Digital Design and Computer Architecture*)
covers this.

---

## Phase 8: TPU Backend — code-complete, hardware-gated

**Start here:** `tpu_engine/README.md` (per-step status table).

| Directory | Topic |
|---|---|
| `gcp_setup/`, `tpu_benchmarks/` | GCP TPU VM provisioning + JAX correctness validation, MXU/HBM/ICI benchmarks |
| `stablehlo_lower/` | MLIR `runtime` dialect → StableHLO lowering pass |
| `stablehlo_execute/` | StableHLO execution via `jax.export`, validated vs. numpy |
| `pjit_distributed/` | pjit-sharded MLP scaling |
| `ici_collectives/` | Gradient all-reduce over ICI |
| `mxu_opt/` | 128-boundary MXU utilization-cliff sweep |
| `vliw_analysis/` | HLO dump + VLIW-vs-OOO-vs-SIMT written analysis |
| `tpu_profiler/` | Combined MXU/HBM/ICI profiler capture |
| `sparsecore/` | SparseCore-vs-dense-gather embedding comparison (v5-only) |
| `layout_opt/`, `hbm_sram/`, `cost_model/` | MXU tile-padding model, HBM↔VMEM overlap model, $/FLOP comparison — **all three actually run locally** (plain `clang++`, no JAX) |

**Primary sources:**
- Jouppi, N. et al. (2017/2023), *"In-Datacenter Performance Analysis of a
  Tensor Processing Unit"* and the TPU v4 follow-up paper — the systolic-
  array MXU architecture `mxu_opt/`, `layout_opt/`, and `vliw_analysis/`
  are all reasoning about.
- Google, *XLA: Optimizing Compiler for Machine Learning* documentation —
  the compilation path StableHLO sits in front of.
- StableHLO project (OpenXLA), *StableHLO specification* — the target
  dialect for `stablehlo_lower/StableHLOLowerPass.cpp`.
- Google/JAX docs — `pjit`, `jax.export`, sharding annotations, used in
  `pjit_distributed/`, `stablehlo_execute/`.

**Background:** the VLIW-vs-out-of-order-superscalar-vs-SIMT comparison in
`vliw_analysis/` assumes the same computer-architecture background as Phase
2/3 (instruction scheduling, hazard hiding); no additional paper needed
beyond the Jouppi TPU papers above.

---

## Phase 9: Inference Serving

**Start here:** `inference_serving/README.md` (per-step status table; most
steps actually run locally).

| Directory | Topic | Locally run? |
|---|---|---|
| `paged_kv/` | Block allocator + per-sequence block table (PagedAttention-style) | yes |
| `continuous_batching/` | Continuous batching vs. static batching | yes |
| `sla_scheduler/` | EDF scheduling with preemption | yes |
| `flash_decoding/` | Split-K decode-step attention (CUDA) | gated |
| `speculative_decoding/` | Draft+verify speculative decoding | yes |
| `gptq/` | Hessian-guided greedy column quantization | yes |
| `kv_quant/` | INT8 K/V cache quantization | yes |
| `serving_backend/` | `ServingRouter` multi-backend dispatch | yes |
| `serving_bench/` | TTFT/TPOT/throughput benchmarking | partial |

**Primary sources:**
- Kwon, W. et al. (2023), *"Efficient Memory Management for Large Language
  Model Serving with PagedAttention"* (vLLM paper) — the block-table KV
  cache design `paged_kv/` implements and the continuous-batching throughput
  finding `continuous_batching/` reproduces as a real measurement.
- Frantar, E. et al. (2022), *"GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers"* — `inference_serving/gptq/` and
  `npu_engine/quant_export/` both cite this directly (Hessian inversion +
  error compensation onto not-yet-quantized columns).
- Leviathan, Y., Kalman, M. & Matias, Y. (2023), *"Fast Inference from
  Transformers via Speculative Decoding"* (and the contemporaneous
  Chen et al. 2023 DeepMind paper) — `speculative_decoding/`'s draft-then-
  verify algorithm and its exactness-vs-plain-greedy-decoding guarantee.
- Dao, T. et al. (2023), *"Flash-Decoding for long-context inference"* —
  the split-K-over-KV-dimension idea `flash_decoding/` targets.

---

## Phase 10: Observability, Testing, AI Integration

**Start here:** `observability/README.md`.

| Directory | Topic | Locally run? |
|---|---|---|
| `ebpf/` | BCC tracepoint/kprobe program | gated (Linux/BCC/sudo) |
| `opentelemetry/` | Hand-rolled span lifecycle + OTLP-JSON export | yes |
| `dashboard/` | Ingestion/histogram/report pipeline | yes |
| `tlc/` | TLC model-checking configs for the Raft/Collective TLA+ specs | gated (Java) |
| `symbiyosys_ci/` | Re-runs Phase 7's formal proofs with change detection | yes |
| `chaos/` | Raft leader-kill + FPGA thermal chaos scenarios | partial |
| `kernel_variant_agent/`, `nsight_agent/`, `llm_autotune/` | Claude-API-driven agent steps (real SDK client code, no API calls made) | portable half only |

**Primary sources:**
- Sigelman, B. et al. (2010), *"Dapper, a Large-Scale Distributed Systems
  Tracing Infrastructure"* — the distributed-tracing model
  `opentelemetry/`'s span/parent-child linkage follows.
- OpenTelemetry project, *OTLP specification* — the exact JSON export format
  `opentelemetry/` parses back and validates against.
- Gregg, B., *BPF Performance Tools* — the standard reference for
  tracepoints/kprobes used in `ebpf/sched_probe.py`.
- Lamport, L. (2002), *Specifying Systems* (again) — TLC model-checking
  usage for `tlc/`.

**Vendor docs:** Anthropic API reference (Messages API, structured
outputs/`output_config.format`) — the client code in the three agent steps
targets this directly.

---

## Phase 12: Machine Learning Library

**Start here:** `ml/README.md`.

### Phase 12a: Classical ML
| Directory | Primary source |
|---|---|
| `decision_tree/` | Breiman, L. et al. (1984), *Classification and Regression Trees* (CART) |
| `random_forest/` | Breiman, L. (2001), *"Random Forests"* |
| `gradient_boosting/` | Friedman, J. (2001), *"Greedy Function Approximation: A Gradient Boosting Machine"*; Friedman, J. (2002), *"Stochastic Gradient Boosting"* (row subsampling) |
| `svm/` | Platt, J. (1998), *"Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines"* |
| `knn/` | Bentley, J.L. (1975), *"Multidimensional Binary Search Trees"* (kd-tree, branch-and-bound); Omohundro, S. (1989), *"Five Balltree Construction Algorithms"*; Uhlmann, J. (1991) (ball trees, independently); Liu, T. et al. (2004) (defeatist-search approximate ball-tree ANN) |
| `kmeans/` | Arthur, D. & Vassilvitskii, S. (2007), *"k-means++: The Advantages of Careful Seeding"* |
| `pca/` | Halko, N., Martinsson, P.G. & Tropp, J. (2011), *"Finding Structure with Randomness"* (randomized SVD) |
| `linear_models/` | Zou, H. & Hastie, T. (2005), *"Regularization and Variable Selection via the Elastic Net"*; Nocedal, J. & Wright, S., *Numerical Optimization*, Algorithm 7.4/7.5 (L-BFGS two-loop recursion) |

### Phase 12b: Decision Framework + Benchmarks
| Directory | Notes |
|---|---|
| `openml_bench/` | Live OpenML REST API, `study/99` = the actual OpenML-CC18 benchmark suite (Bischl, B. et al. (2019), *"OpenML Benchmarking Suites"*) |
| `cross_method/`, `decision_criteria/`, `ensemble/`, `failure_modes/`, `hyperparam_sensitivity/` | Written analyses grounded in `openml_bench`'s real numbers; no additional external citation beyond the algorithms' own papers above |

### Phase 12c: Hyperparameter Optimization
| Directory | Primary source |
|---|---|
| `bayesian_opt/` | Rasmussen, C.E. & Williams, C.K.I. (2006), *Gaussian Processes for Machine Learning*; Mockus, J. (1978), *"The Application of Bayesian Methods for Seeking the Extremum"* (Expected Improvement) |
| `tpe/` | Bergstra, J. et al. (2011), *"Algorithms for Hyper-Parameter Optimization"* (Tree-structured Parzen Estimator) |
| `hyperband/` | Jamieson, K. & Talwalkar, A. (2016), *"Non-stochastic Best Arm Identification and Hyperparameter Optimization"* (Successive Halving); Li, L. et al. (2016), *"Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"*; Li, L. et al. (2018), *"Massively Parallel Hyperparameter Tuning"* (ASHA) |
| `pbt/` | Jaderberg, M. et al. (2017), *"Population Based Training of Neural Networks"* |
| all four | Bergstra, J. & Bengio, Y. (2012), *"Random Search for Hyper-Parameter Optimization"* — the "random search is a strong baseline" finding all four steps independently reproduce on this repo's real OpenML tasks |

---

## Phase 13: Retrieval-Augmented Generation

**Start here:** `rag/DESIGN.md`.

| Directory | Topic |
|---|---|
| `embedding_model/` | Bidirectional encoder, InfoNCE/CLIP-style contrastive loss |
| `hnsw/` (also `ml/knn/`'s `CosineBallTree`) | HNSW vs. ball-tree ANN benchmark |
| `indexing_pipeline/` | Sentence-boundary-aware chunking + embed + index |
| `rag_generation/` | Prompt construction + generation |
| `recall_eval/` | recall@k measurement, exact vs. approximate |
| `generation_quality/` | Retrieval-conditioned generation accuracy |
| `approx_retrieval_study/` | Composes recall drop → generation-accuracy cost |
| `serving_integration/` | `route_rag()` — non-invasive `ServingRouter` integration |

**Primary sources:**
- Malkov, Y. & Yashunin, D. (2016/2018), *"Efficient and Robust Approximate
  Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"*
  — `rag/hnsw/hnsw.h` implements this directly (Algorithm 2, SEARCH-LAYER,
  cited by name).
- van den Oord, A., Li, Y. & Vinyals, O. (2018), *"Representation Learning
  with Contrastive Predictive Coding"* — the InfoNCE loss
  `embedding_model/embedding_model.h` cites directly.
- Radford, A. et al. (2021), *"Learning Transferable Visual Models From
  Natural Language Supervision"* (CLIP) — the symmetric query↔document
  contrastive formulation `embedding_model.h` also cites directly.
- Lewis, P. et al. (2020), *"Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks"* — the RAG paradigm the whole phase
  implements (retrieve-then-generate), background for `rag_generation/`.

---

## Phase 14: Adversarial Robustness

**Start here:** `adversarial/DESIGN.md`.

| Directory | Primary source |
|---|---|
| `input_gradients/` | Foundational — no new paper, verifies the existing autograd tape's `x.grad()` via finite differences |
| `fgsm/` | Goodfellow, I., Shlens, J. & Szegedy, C. (2014), *"Explaining and Harnessing Adversarial Examples"* |
| `pgd/` | Madry, A. et al. (2017), *"Towards Deep Learning Models Resistant to Adversarial Attacks"* |
| `vulnerability_measurement/` | Composes FGSM/PGD against the undefended model |
| `adversarial_training/` | Madry et al. (2017) again — the min-max training formulation |
| `robustness_tradeoff/` | Tsipras, D. et al. (2018), *"Robustness May Be at Odds with Accuracy"* |
| `transferability/` | Szegedy, C. et al. (2013), *"Intriguing Properties of Neural Networks"*; Papernot, N. et al. (2016), *"Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples"* |
| `randomized_smoothing/` | Cohen, J., Rosenfeld, E. & Kolter, Z. (2019), *"Certified Adversarial Robustness via Randomized Smoothing"* |

Note: `randomized_smoothing/randomized_smoothing.h` documents one disclosed
simplification vs. the paper — a normal/Wald confidence bound in place of
Cohen et al.'s exact Clopper-Pearson bound.

**Background:** Clopper, C.J. & Pearson, E.S. (1934), *"The Use of
Confidence or Fiducial Limits Illustrated in the Case of the Binomial"* — the
exact bound Cohen et al. use and this repo's Wald approximation stands in
for.

---

## Phase 15: NPU Backend

**Start here:** `npu_engine/DESIGN.md`.

| Directory | Topic |
|---|---|
| `quant_export/` | Reuses `inference_serving::GptqQuantizer` (Frantar et al. 2022, see Phase 9) at INT8, custom `.npuw` format |
| `cost_model/` | Portable INT8 latency/efficiency model (NPU vs. GPU) |
| `op_coverage/` | Walks all 18 `runtime` dialect ops, finds gather/scatter unsupported |
| `thermal/` | Mirrors `fpga_engine/thermal_router`'s decision/hardware-read split |

**Vendor docs:** Apple *Core ML* documentation (ANE targeting), ONNX Runtime
*NPU execution providers* documentation — both referenced as the
still-unrun `npu_onnx_export.py` path (no `coremltools`/`onnx`/
`onnxruntime` installed locally; see project memory on the standing
no-new-local-installs decision).

---

## Phase 16: Containerized & Orchestrated Deployment

**Start here:** `containers/DESIGN.md`, `containers/README.md`.

| Artifact | Topic |
|---|---|
| `Dockerfile` | Multi-stage build, portable subset |
| `docker/gpu/` | GPU passthrough (`daemon.json`, `Dockerfile.cuda`, compose) |
| `k8s/serving/` | Latency-driven HPA (not raw CPU%) for `serving_daemon` |
| `k8s/training/` | `StatefulSet` with `podManagementPolicy: Parallel` for gang-scheduled training, targets `distributed_training/training_worker/` |

**Vendor docs:** Docker docs (multi-stage builds, `nvidia-container-toolkit`
/ NVIDIA Container Runtime docs for `docker/gpu/`), Kubernetes docs
(`HorizontalPodAutoscaler` custom-metrics API, `StatefulSet` semantics
including `podManagementPolicy`).

---

## Phase 17: Analog & Unconventional Compute Hardware — scoped, not yet built

**Start here:** none yet — `analog_engine/DESIGN.md` will exist once step 1
lands. Until then, PLAN.md's Phase 17 section is the closest thing to a
"start here."

| Planned step | Topic |
|---|---|
| 1 | RRAM-like device non-ideality model (read/write noise, drift, endurance) |
| 2 | Resistive crossbar analog MAC simulation |
| 3 | Non-volatile memory tradeoff comparison (RRAM/PCM/STT-MRAM/SRAM-CIM) |
| 4 | Analog-vs-digital MAC energy model |
| 5 | From-scratch PPA/dataflow model (WS/OS/IS/RS) |
| 6 | Systolic array design-space sweep |
| 7 | Hardware-algorithm co-design case study |
| 8 | Analog circuit transient (RC step-response) surrogate |

**Primary sources:**
- Shafiee, A. et al. (2016), *"ISAAC: A Convolutional Neural Network
  Accelerator with In-Situ Analog Arithmetic in Crossbars"* — the canonical
  resistive-crossbar-as-MAC-engine architecture step 2 simulates.
- Chi, P. et al. (2016), *"PRIME: A Novel Processing-in-Memory Architecture
  for Neural Network Computation in ReRAM-Based Main Memory"* — a second
  foundational crossbar-PIM architecture, useful contrast to ISAAC's design
  choices.
- Yu, S. (2018), *"Neuro-Inspired Computing with Emerging Nonvolatile
  Memory"* (Proceedings of the IEEE) — the device-physics review step 1's
  noise/drift/endurance model and step 3's NVM comparison table draw on.
- Sze, V., Chen, Y-H., Yang, T-J. & Emer, J. (2017), *"Efficient Processing
  of Deep Neural Networks: A Tutorial and Survey"* — the canonical
  reference for the Weight-/Output-/Input-/Row-Stationary dataflow
  taxonomy step 5 implements.
- Chen, Y-H., Emer, J. & Sze, V. (2016/2017), *"Eyeriss: A Spatial
  Architecture for Energy-Efficient Dataflow for Convolutional Neural
  Networks"* — the concrete Row-Stationary accelerator design step 5/6
  are modeled after.
- Parashar, A. et al. (2019), *"Timeloop: A Systematic Approach to DNN
  Accelerator Evaluation"* — the tool step 5 reproduces a minimal version
  of (workload + architecture description → utilization/data-movement/
  energy), without depending on installing it.
- Wu, Y.N., Emer, J. & Sze, V. (2019), *"Accelergy: An Architecture-Level
  Energy Estimation Methodology for Accelerator Designs"* — the companion
  energy-estimation methodology step 4/5 mirror the shape of.
- Ambrogio, S. et al. (2018), *"Equivalent-accuracy accelerated
  neural-network training using analog memory"* (Nature) — a real
  measured analog-training result, useful ground-truth calibration for
  step 2's accuracy-vs-noise curve.

**Background:** basic circuit theory (Ohm's law, Kirchhoff's current law)
is the entire mathematical content behind "a resistive crossbar computes a
matrix-vector product for free" — no deeper circuit-design text is needed
before step 2; Sze et al. 2017 above is also the right on-ramp for anyone
who hasn't seen the dataflow-taxonomy material before.

---

## Phase 18: Dynamical Systems, SciML & Physics-Informed Architectures — scoped, not yet built

**Start here:** none yet — `sciml/DESIGN.md` will exist once step 1 lands.

| Planned step | Topic |
|---|---|
| 1 | ODE solver library (explicit Euler, RK4, implicit/backward Euler) |
| 2 | Stability & stiffness analysis |
| 3 | SDE solver (Euler-Maruyama, Milstein) |
| 4 | Neural ODEs (adjoint sensitivity) |
| 5 | Deep Equilibrium Models |
| 6 | State-space model layer vs. attention |
| 7 | Diffusion / flow-matching generative model |
| 8 | Energy-based model |
| 9 | muP-style scaling study |
| 10 | Physics-informed / noise-aware training (bridges to Phase 17 step 1) |

**Primary sources:**
- Chen, R.T.Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. (2018),
  *"Neural Ordinary Differential Equations"* (NeurIPS, best paper) — step
  4's entire approach: parameterize `dx/dt` with a network, backprop via
  the adjoint method instead of unrolling the solver.
- Pontryagin, L.S. et al. (1962), *The Mathematical Theory of Optimal
  Control Processes* — the classical origin of the adjoint/costate method
  Chen et al. (2018) repurpose for backprop; useful for understanding
  *why* the adjoint ODE gives the right gradient, not just that it does.
- Kloeden, P.E. & Platen, E. (1992), *Numerical Solution of Stochastic
  Differential Equations* — the standard reference for Euler-Maruyama/
  Milstein (step 3) and their convergence-order guarantees.
- Bai, S., Kolter, J.Z. & Koltun, V. (2019), *"Deep Equilibrium Models"*
  (NeurIPS) — step 5's fixed-point-layer formulation and
  implicit-function-theorem backprop.
- Broyden, C.G. (1965), *"A Class of Methods for Solving Nonlinear
  Simultaneous Equations"* — the quasi-Newton fixed-point solver step 5
  can use in place of plain fixed-point iteration.
- Gu, A., Goel, K. & Ré, C. (2021), *"Efficiently Modeling Long Sequences
  with Structured State Spaces"* (S4) — step 6's linear state-space
  recurrence layer.
- Gu, A. & Dao, T. (2023), *"Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces"* — the input-dependent (selective) extension of
  S4, background for step 6's writeup even if the implemented layer stays
  closer to plain S4 for tractability.
- Sohl-Dickstein, J. et al. (2015), *"Deep Unsupervised Learning using
  Nonequilibrium Thermodynamics"* — the original diffusion-model
  formulation.
- Ho, J., Jain, A. & Abbeel, P. (2020), *"Denoising Diffusion Probabilistic
  Models"* (DDPM) — the practical forward-noising/learned-reverse-process
  formulation step 7 implements.
- Lipman, Y. et al. (2023), *"Flow Matching for Generative Modeling"* — the
  continuous-normalizing-flow alternative to diffusion step 7 optionally
  also implements.
- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. & Huang, F. (2006), *"A
  Tutorial on Energy-Based Learning"* — step 8's EBM formulation and
  contrastive-divergence/score-matching training.
- Hinton, G. (2002), *"Training Products of Experts by Minimizing
  Contrastive Divergence"* — the specific contrastive-divergence training
  procedure step 8 can use.
- Yang, G. & Hu, E.J. et al. (2021/2022), *"Tensor Programs V: Tuning
  Large Neural Networks via Zero-Shot Hyperparameter Transfer"* (muP) —
  step 9's entire claim under test: hyperparameters tuned at small width
  transfer to large width under the right parameterization.

**Background:** an ODE/PDE course covering explicit-vs-implicit solvers
and stability (any standard numerical-methods text, e.g. Iserles, *A First
Course in the Numerical Analysis of Differential Equations*) is the right
on-ramp before step 1; Goodfellow, Bengio & Courville's *Deep Learning*
ch. 20 (generative models) is useful connective tissue between steps 7/8
if the primary sources above move too fast.

---

## Phase 19: Framework-Native Training (PyTorch/JAX) — scoped, not yet built

**Start here:** none yet — `framework_native/DESIGN.md` will exist once
step 1 lands.

| Planned step | Topic |
|---|---|
| 1 | PyTorch port of `transformer/` |
| 2 | `torch.compile` benchmarking |
| 3 | DDP over CPU (`gloo`) |
| 4 | FSDP vs. hand-written ZeRO |
| 5 | JAX port (`jit`/`grad`/`vmap`/`pmap`) |
| 6 | One real run through a production framework |

**Primary sources:**
- Paszke, A. et al. (2019), *"PyTorch: An Imperative Style,
  High-Performance Deep Learning Library"* — the autograd/eager-execution
  design step 1 targets directly.
- Li, S. et al. (2020), *"PyTorch Distributed: Experiences on Accelerating
  Data Parallel Training"* — the real DDP paper (gradient bucketing,
  overlapping backward with all-reduce) step 3 is built on and compared
  against `distributed_training/data_parallel`.
- Zhao, Y. et al. (2023), *"PyTorch FSDP: Experiences on Scaling Fully
  Sharded Data Parallel"* — step 4's target, and the direct paper-level
  counterpart to compare against `zero1`/`zero2`/`zero3`'s own design
  (which independently implements Rajbhandari et al.'s ZeRO, see Phase 6
  above — FSDP is essentially ZeRO stage 3 as a first-class PyTorch API).
- Bradbury, J. et al. (2018), *JAX: composable transformations of
  Python+NumPy programs* (software, with an accompanying design writeup)
  — the `jit`/`grad`/`vmap`/`pmap` functional-transform model step 5
  targets; already partially in use for `tpu_engine`.
- PyTorch documentation: *TorchDynamo/TorchInductor* (`torch.compile`
  internals) — needed to interpret step 2's speedup-or-not result rather
  than treat it as a black box.
- Documentation for whichever of PyTorch Lightning / DeepSpeed / Ray Train
  ends up used in step 6 — Rasley, J. et al. (2020), *"DeepSpeed: System
  Optimizations Enable Training Deep Learning Models with Over 100 Billion
  Parameters"* if DeepSpeed; Moritz, P. et al. (2018), *"Ray: A
  Distributed Framework for Emerging AI Applications"* if Ray Train.

**Background:** none beyond what Phase 6 already assumes (backprop,
optimizers, data/tensor/pipeline parallelism) — this phase is explicitly
about mapping already-understood concepts onto specific framework APIs,
not new theory.

---

## Cross-cutting: testing & methodology

These apply across every phase, not to one:

- **Property-based testing** (`foundation/proptest`) — Claessen & Hughes
  (2000), *QuickCheck* (already listed under Phase 1); reused directly by
  `inference_serving/paged_kv`'s 300-trial property test.
- **TSan/ASan/UBSan discipline** — every concurrent structure must be race-
  clean before a step counts as done (see project memory:
  `feedback_tsan_discipline`). Background: ThreadSanitizer's algorithm is
  Serebryany, K. & Iskhodzhanov, T. (2009), *"ThreadSanitizer: Data Race
  Detection in Practice"*.
- **Formal verification** (`fpga_engine/symbiyosys`) — k-induction model
  checking; Sheeran, M., Singh, S. & Stålmarck, G. (2000), *"Checking Safety
  Properties Using Induction and a SAT-Solver"* is the standard reference
  for the k-induction technique SymbiYosys's `mode prove` uses.
- **Roofline modeling** — Williams et al. 2009 (already listed under Phase
  2), reused verbatim in Phase 3 and structurally in Phase 8's MXU/HBM
  analysis.

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
| GPU | CUDA C++ Programming Guide, PTX ISA, NVML API reference, Nsight Compute/Systems docs, Hopper Architecture Whitepaper |
| Compiler | MLIR language reference, TableGen ODS reference, LLVM IR reference manual |
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
