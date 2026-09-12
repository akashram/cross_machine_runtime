# Repo Review Plan — 15-30 min/day

Goal: get to "can explain it out loud, unscripted" on every phase of
`cross_machine_runtime`, without ever spending more than 30 minutes in one
sitting. **257 days total** (~1 year at 5x/week, less if you go 7x/week or
double up on light days). Order follows the repo's own build order
(Phase 1 → 19), so later days' "why it matters" notes assume you've seen
the earlier phases.

**Daily routine (pick whichever fits the day's entry):**
1. Read that step's `README.md` — findings/results first, "what was built"
   second.
2. Skim the primary source file for ~10-15 min — you're looking for the
   shape of the solution, not memorizing it.
3. Write one sentence in a running log: what surprised you, or what you'd
   say if someone asked "what does this do and why."
4. If the step is one of the ones actually run locally (marked below) and
   you have 5 spare minutes, `ctest -R <name>` and watch it pass — seeing
   real output sticks better than reading about it.

Most steps here are hardware-gated and unrun (no GPU/FPGA/TPU/Linux on
this Mac) — for those, you're reading real code and a documented finding,
not re-running anything. That's fine; the goal is recognition-level
fluency, not reimplementation.

---

## Day 0 — Orientation
Read CLAUDE.md's "Where we are" section top to bottom (skim, don't study),
plus PLAN.md and SCOPE.md's tables of contents. Just get the shape of the
19 phases before diving into any one of them.

---

## Phase 1: Foundation (`foundation/`) — Days 1-15
Lock-free data structures and allocators everything else is built on.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 1 | `aba` | The ABA problem in lock-free CAS | Why a pointer looking "unchanged" doesn't mean the memory underneath wasn't reused |
| 2 | `arena` | Bump-pointer arena allocator | Fast allocation, no per-object free — the tradeoff vs malloc |
| 3 | `freelist` | Lock-free freelist | Real TSan bug: `next` pointer and `storage` shared a union and raced; fixed by separating them |
| 4 | `chase_lev` | Chase-Lev work-stealing deque | Real TSan bug: a fence+relaxed-store pattern TSan doesn't model; fixed via release-store on `bottom_` |
| 5 | `ws_pool` | Work-stealing pool built on chase_lev | 3 real TSan bugs in one component — ctor race, lost wakeup, and a fence false-positive |
| 6 | `coro` | Coroutine engine | The scheduling primitive several later "process-per-rank" and serving steps build on |
| 7 | `epoch` | Epoch-based reclamation | Deferred free until no reader could still be using an object |
| 8 | `hazard` | Hazard pointers | A second reclamation scheme — compare its tradeoffs against epoch |
| 9 | `rcu` | Read-Copy-Update | A third reclamation scheme, read-mostly optimized — you now have 3 to compare |
| 10 | `msqueue` | Michael-Scott lock-free queue | The canonical CAS-loop queue algorithm from the original paper |
| 11 | `numa` | NUMA-aware allocation | Placement matters even before you get to distributed systems |
| 12 | `perf` | Hardware counter infrastructure | The IPC/cache-miss plumbing every later benchmark in the repo depends on |
| 13 | `proptest` | Property-based testing framework | Generates random operation sequences to fuzz the structures above |
| 14 | `tensor` | Tensor handle abstraction | The shape/stride/device concept reused all the way through the compiler, GPU, and TPU phases |
| 15 | *(synthesis)* | Re-skim `feedback_tsan_discipline` findings | Write one sentence: which reclamation scheme (epoch/hazard/RCU) would you pick for a read-heavy cache, and why |

---

## Phase 2: CPU Backend (`cpu_engine/`) — Days 16-29

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 16 | `affinity` | CPU affinity pinning | Reduces cache-line bouncing between cores |
| 17 | `hugepage` | Huge pages | Fewer TLB misses on large allocations |
| 18 | `os_tuning` | OS-level tuning knobs | Interrupt affinity / scheduler settings for latency-sensitive work |
| 19 | `prefetch` | Software prefetching | Hiding memory latency ahead of use |
| 20 | `branchless` | Branchless programming | Avoiding misprediction penalties |
| 21 | `avx512` | AVX-512 SIMD | Needs a Linux/AVX-512 box — unrun here (this Mac is AVX2-only) |
| 22 | `nt_store` | Non-temporal stores | Bypassing cache pollution on streaming writes |
| 23 | `tiling` | Cache/register tiling | The CPU-side precursor to GPU shared-memory tiling and TPU MXU tiling you'll see later |
| 24 | `roofline` | Roofline model | Compute-bound vs. memory-bound classification — reused conceptually through TPU and analog phases |
| 25 | `inference` | CPU inference engine | Assembles the techniques above into a real forward pass |
| 26 | `perf_deep_dive` | Deeper hardware counter analysis | Goes beyond `foundation/perf`'s basics |
| 27 | `pgo` | Profile-guided optimization | Compiler feedback loop using real execution profiles |
| 28 | `busy_poll` | Busy-poll vs. blocking receive | Latency-critical polling tradeoff |
| 29 | *(synthesis)* | — | After today you should be able to explain roofline analysis in 2 sentences — you'll see it again in `gpu_engine/roofline` and `tpu_engine` |

---

## Phase 3: GPU Backend (`gpu_engine/`) — Days 30-51
All hardware-gated/unrun (no CUDA toolchain locally) except the two
written comparisons in steps 25-26.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 30 | `device_query` | Baseline device query | What a GPU reports about itself before anything else runs |
| 31 | `memory` | CUDA memory management | Pinned memory, unified memory tradeoffs |
| 32 | `streams` | CUDA streams | Overlapping compute/copy via async execution |
| 33 | `warp_primitives` | Warp-level primitives | SIMT's answer to cross-lane communication (shuffle/vote/ballot) |
| 34 | `shared_mem` | Shared-memory tiling | The GPU analog of `cpu_engine/tiling` |
| 35 | `coalescing` | Memory coalescing | Access pattern determines effective bandwidth |
| 36 | `occupancy` | Occupancy | Register/shared-mem pressure vs. how many warps hide latency |
| 37 | `kernels` | Elementwise + GEMM kernels | The actual compute kernels the rest of the phase optimizes |
| 38 | `ptx_sass` | PTX/SASS inspection | Reading what the compiler actually generated |
| 39 | `flash_attn` | Flash attention | Tiled, IO-aware attention — the algorithm `inference_serving/flash_decoding` later specializes |
| 40 | `graphs` | CUDA graphs | Capturing a kernel sequence to cut launch overhead |
| 41 | `p2p` | GPU-to-GPU peer access | Foundation for multi-GPU without going through host |
| 42 | `precision` | Mixed precision / FP8 | Numerics tradeoffs for training/inference |
| 43 | `sparsity` | 2:4 structured sparsity | Hardware-accelerated sparse matmul on Ampere+ |
| 44 | `hopper` | Hopper TMA/WGMMA | The newest tensor-core programming model, one gen past `kernels`' WMMA |
| 45 | `roofline` | GPU roofline | Same model as CPU, applied once real hardware numbers exist |
| 46 | `power` | NVML power monitoring | Reading real power draw off the device |
| 47 | `mps` | Multi-Process Service | Sharing one GPU across processes cheaply |
| 48 | `nsight_ci` | Nsight profiling in CI | The automated half of this phase |
| 49 | `triton_kernels` | Step 25 — Triton reimplementation | Read the written comparison: what Triton's block-level model abstracts away vs. hand-written CUDA, and what it doesn't |
| 50 | `cutlass_gemm` | Step 26 — CUTLASS SM80 GEMM | Compared against `kernels`' hand-written WMMA GEMM |
| 51 | *(synthesis)* | `gpu_engine/DESIGN.md` §7 | None of Phase 3 has run on real hardware — this section is the one part you can fully evaluate as writing alone |

---

## Phase 4: Compiler/MLIR (`compiler/`) — Days 52-66
Toolchain-gated (needs an LLVM/MLIR source build on Linux); `cost_model`
is the one exception, actually built and run with plain `clang++`.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 52 | `dialect` | The custom `runtime` MLIR dialect | 15 ops, 3 attrs — everything else in this phase transforms this IR |
| 53 | `mlir_setup` | LLVM/MLIR build setup | The toolchain gate itself |
| 54 | `shape_inference` | Shape inference pass | Propagates tensor shapes through the IR |
| 55 | `fusion` | Fusion pass | Merges ops to cut memory round-trips |
| 56 | `affine_lower` | Affine lowering/tiling | High-level ops → affine loop nests |
| 57 | `mem_planning` | Memory planning pass | Static buffer allocation across the graph |
| 58 | `remat` | Rematerialization pass | Recompute-vs-store tradeoff for activations |
| 59 | `placement` | Placement pass | Decides which device runs which op — the file Phase 8 and Phase 15 later edit directly |
| 60 | `sharding` | Auto-sharding pass | Splits ops across devices automatically |
| 61 | `kernel_spec` | Kernel specialization pass | Picks a concrete kernel implementation per op |
| 62 | `aot` | AOT pipeline | Orchestrates every pass above + LLVM codegen + link |
| 63 | `cost_model` | **Actually run locally** | Read its README's real captured output — your only executable evidence in this phase |
| 64 | `fuzzing` | Dialect/pass fuzzing harness | — |
| 65 | `upstream` | Tracking-upstream notes | — |
| 66 | *(synthesis)* | Trace the `placement` edit from `npu_engine` | Phase 4 gets touched twice more, in Phase 8 and Phase 15 — see how |

---

## Phase 5: Distributed/Networking (`networking/`) — Days 67-93
The phase with the repo's best debugging story. Mostly run locally
(real TCP/threads, no EFA/Linux dependency for 14 of 26 components).

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 67 | `common` | The shared `Channel` transport | Real POSIX sockets — everything else in this phase is built on it |
| 68 | `rdma_v1` | TCP baseline | The "before" picture for the RDMA comparison |
| 69 | `efa_setup` | EFA setup | AWS's RDMA-like fabric, hardware-gated |
| 70 | `efa_srd` | EFA/SRD | Code-complete, hardware-gated |
| 71 | `rdma_onesided` | One-sided RDMA verbs | Put/get without remote CPU involvement |
| 72 | `ptp` | Precision Time Protocol | Clock sync across nodes |
| 73 | `grpc_control` | gRPC control plane | — |
| 74 | `flatbuffers_data` | FlatBuffers data plane | Zero-copy serialization for the hot path |
| 75 | `af_xdp` | AF_XDP kernel-bypass networking | Linux-only |
| 76 | `userspace_net` | Userspace networking | Companion to af_xdp |
| 77 | `nic_deep_dive` | NIC internals | — |
| 78 | `ring_allreduce` | Ring all-reduce | Had a real caught chunk-ownership off-by-one bug |
| 79 | `halving_doubling` | Halving-doubling all-reduce | Different bandwidth/latency tradeoff than ring |
| 80 | `tree_allreduce` | Tree all-reduce | Third algorithm — compare all 3 |
| 81 | `collectives` | Broadcast/reduce-scatter/all-gather | Built on the 3 all-reduce algorithms above |
| 82 | `nccl_tuning` | NCCL tuning config | — |
| 83 | `topo_scheduler` | Topology-aware scheduler | Places comm-heavy work based on network topology |
| 84 | `vector_clocks` | Vector clocks | Causal ordering without a global clock |
| 85 | `chandy_lamport` | Distributed snapshots | Never had the detach-not-join bug below — because it joins via a real mutual-shutdown protocol. Read right before `raft` |
| 86 | `raft` | Leader election + log replication | **Read the full README bug timeline** — a real SIGSEGV use-after-free (detach-not-join), then a second bug the fix itself exposed. The best debugging case study in the repo |
| 87 | `tla_raft` | TLA+ spec for Raft | Java/TLC-gated formal verification |
| 88 | `tla_collective` | TLA+ spec for the collective protocol | — |
| 89 | `backpressure` | Credit-based flow control | Redesigned after root-causing a flaky test instead of widening its slack constant; also had raft's exact bug via a stale comment |
| 90 | `hedged_requests` | Hedged requests | Intentional `.detach()` (unlike backpressure's bug) — plus a real shared-RNG TSan race in its own test |
| 91 | `multitenancy` | Multi-tenant fairness | — |
| 92 | `chaos` | Chaos engineering harness | The general fault-injection tool `observability/chaos` reuses later |
| 93 | *(synthesis)* | — | Write 2 sentences: the common shape between raft's bug and backpressure's bug, and why chandy_lamport never had it |

---

## Phase 6: Distributed GPU Training (`distributed_training/` + `transformer/`) — Days 94-121

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 94 | `data_loading` | Data loading pipeline | — |
| 95 | `data_parallel` | Basic data parallelism | The baseline every later parallelism strategy is compared against |
| 96 | `grad_accum` | Gradient accumulation | Simulating larger batches without more memory |
| 97 | `grad_clipping` | Gradient clipping | Numerical stability guard |
| 98 | `autograd` | Hand-rolled reverse-mode autograd + toy MLP | The exact tape `adversarial/` attacks in Phase 14 |
| 99 | `zero1` | ZeRO stage 1 | Shards optimizer state |
| 100 | `zero2` | ZeRO stage 2 | Also shards gradients |
| 101 | `zero3` | ZeRO stage 3 | Also shards parameters |
| 102 | `zero_infinity` | ZeRO-Infinity | Offloads to CPU/NVMe beyond ZeRO-3 |
| 103 | `col_row_linear` | Column/row-parallel linear | Tensor parallelism's basic building block |
| 104 | `tensor_parallel_attn` | Tensor-parallel attention | col_row_linear applied to attention |
| 105 | `seq_parallel` | Sequence parallelism | Splits along the sequence dimension instead |
| 106 | `pipeline_1f1b` | 1F1B pipeline scheduling | Interleaves forward/backward across stages |
| 107 | `parallel_3d` | 3D parallelism | Data + tensor + pipeline combined |
| 108 | `moe` | Mixture-of-experts | Expert parallelism |
| 109 | `checkpoint` | Checkpoint sharding | Across ranks |
| 110 | `compute_comm_overlap` | Compute/comm overlap | Hides collective latency behind compute |
| 111 | `sync_batchnorm` | SyncBatchNorm | Batch-norm stats synced across data-parallel ranks |
| 112 | `full_training_loop` | The full assembled training loop | The shared skeleton for the SFT/reward/PPO/DPO steps that follow |
| 113 | `sparsity_training` | 2:4 sparsity during training | Not just inference this time |
| 114 | `sft` | Supervised fine-tuning | Trains the real transformer, perplexity 18.3→1.07 |
| 115 | `reward_model` | Bradley-Terry reward model | Real overfitting bug found: held-out-prompt split collapsed accuracy to 0%; fixed by splitting at the pair level |
| 116 | `ppo_rlhf` | PPO-based RLHF | Reward rises while KL stays bounded — no reward-hacking signature |
| 117 | `dpo` | Direct Preference Optimization | Reuses reward_model's exact loss unchanged (Rafailov's reparameterization) — compare against ppo_rlhf |
| 118 | `gpudirect_storage` | GPUDirect Storage | The one hardware-gated step with no portable subset |
| 119 | `training_worker` | Real process-per-rank driver | Validated as 4 actual OS processes, not simulated threads |
| 120 | `transformer/` | The minimal decoder-only transformer | Built specifically so steps 22-25 have a real model to train |
| 121 | *(synthesis)* | — | sft → reward_model → ppo_rlhf → dpo is one continuous story on one model. One sentence: why doesn't DPO need a separate reward model? |

---

## Phase 7: FPGA Backend (`fpga_engine/`) — Days 122-147
Mostly hardware-gated; `cocotb` and `symbiyosys` are actually run.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 122 | `f1_setup` | F1 instance validation baseline | — |
| 123 | `tcl_pipeline` | TCL synth/impl/bitstream pipeline | — |
| 124 | `power_ci` | Vivado power report in CI | — |
| 125 | `axi_stream` | AXI4-Stream passthrough | The RTL reused by cocotb, symbiyosys, and ila_debug |
| 126 | `dot_product` | Initiation-interval study | — |
| 127 | `loop_opt` | UNROLL/PIPELINE/DATAFLOW comparison | — |
| 128 | `dsp_lut` | DSP48E2 vs LUT tradeoff | — |
| 129 | `fixed_point` | ap_fixed precision/resource/latency study | — |
| 130 | `bram_uram` | BRAM vs URAM access patterns | — |
| 131 | `ddr4` | Multi-bank DDR4 integration | — |
| 132 | `dma` | Host-side DMA via XRT | — |
| 133 | `pcie_latency` | PCIe latency decomposition | — |
| 134 | `pingpong` | Double-buffered compute/transfer overlap | — |
| 135 | `ml_kernel` | Pipelined INT8 MLP kernel | Reused by `vitis_ai`'s comparison later |
| 136 | `timing_closure` | Critical-path analysis + retiming | Its cycle-count model is cited directly by `vitis_ai` |
| 137 | `slr` | SLR partitioning + crossing penalty | — |
| 138 | `clock_gating` | Dynamic-power modeling | — |
| 139 | `xadc` | Real XRT thermal/electrical sensor API | Unrun, but the portable JSON parser is tested |
| 140 | `ila_debug` | ILA debug core on AXI4-Stream | Portable checker catches a synthetic protocol bug |
| 141 | `cocotb` | **Actually run** — testbenches vs Icarus Verilog | Caught a real one-cycle-early memory-read bug in the DMA controller |
| 142 | `symbiyosys` | **Actually run** — formal proofs on the same RTL | Caught 2 formal-*harness* bugs (not RTL bugs) — read right after cocotb for the test-vs-proof contrast |
| 143 | `partial_reconfig` | DFX hot-swap flow | Two interface-compatible kernels in one reconfigurable pblock |
| 144 | `fpga_net` | P4 RDMA-like bypass path | Portable model predicts 13.7x speedup vs CPU-mediated networking |
| 145 | `vitis_ai` | DPU vs. hand-written kernel | Caught a real printf/varargs UB bug along the way |
| 146 | `thermal_router` | Thermal-aware allocation policy | Shares decision logic between the real and simulated paths so they can't diverge |
| 147 | *(synthesis)* | `fpga_engine/README.md` status table | cocotb and symbiyosys are your only actually-measured FPGA results — name the 3 steps closest to unblocked given real hardware |

---

## Phase 8: TPU Backend (`tpu_engine/`) — Days 148-161
`layout_opt`, `hbm_sram`, `cost_model` are actually run locally; the rest
need real TPU/JAX hardware.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 148 | `gcp_setup` | TPU VM provisioning + JAX validation | — |
| 149 | `tpu_benchmarks` | MXU/HBM/ICI benchmark scripts | — |
| 150 | `stablehlo_lower` | Runtime dialect → StableHLO pass | Covers every op except gather/scatter — a disclosed real gap |
| 151 | `stablehlo_execute` | Runs lowered StableHLO via jax.export | Checked against a numpy reference |
| 152 | `pjit_distributed` | pjit-sharded MLP scaling | Mirrors distributed_training's column/row sharding for a cross-backend comparison |
| 153 | `ici_collectives` | Gradient all-reduce over ICI | — |
| 154 | `mxu_opt` | 128-boundary utilization-cliff sweep | Independently reproduced by `analog_engine/systolic_sweep` from the opposite direction |
| 155 | `vliw_analysis` | VLIW-vs-OOO-vs-SIMT written analysis | The load-bearing point: a bad TPU bundle-packing decision has no hardware fallback |
| 156 | `tpu_profiler` | Combined MXU/HBM/ICI profiler capture | — |
| 157 | `sparsecore` | SparseCore-vs-dense-gather comparison | v5-only; raises `NotImplementedError` honestly rather than faking the API |
| 158 | `layout_opt` | **Actually run** — MXU tile-padding ceiling | batch=1 decode hits 0.8% ceiling — argues for batching decode regardless of hardware |
| 159 | `hbm_sram` | **Actually run** — HBM↔VMEM overlap model | Explains XLA's preference for cubic tiles |
| 160 | `cost_model` | **Actually run** — TPU/A100/H100 $/FLOP | All land within ~5% at peak — utilization ceiling, not peak spec, is the real device-choice driver |
| 161 | *(synthesis)* | — | Connect `mxu_opt`'s finding to `analog_engine/systolic_sweep`'s — same phenomenon, two hardware types |

---

## Phase 9: Inference Serving (`inference_serving/`) — Days 162-171
Almost entirely actually run locally.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 162 | `paged_kv` | Block-based KV cache allocator | Property-tested, 300 trials |
| 163 | `continuous_batching` | Discrete-event sim vs static batching | 1.85x throughput — reproduces the vLLM paper's finding as a real measurement |
| 164 | `sla_scheduler` | Earliest-Deadline-First scheduling | 0% SLA violations vs FIFO's 36.5% |
| 165 | `flash_decoding` | Split-K decode attention | The one hardware-gated step (needs a parallelism axis single-row decode has and prefill doesn't) |
| 166 | `speculative_decoding` | Draft+verifier spec decode | Proves output is token-identical to plain greedy regardless of draft quality — a correctness proof, not just a speedup number |
| 167 | `gptq` | Hessian-guided quantization | Real segfault caught from a weight-layout mismatch during development |
| 168 | `kv_quant` | INT8 K/V quantization | Exactly 4.000x memory reduction, measured |
| 169 | `serving_backend` | ServingRouter with honest `available=false` | Tests the fallback priority order itself, not just "falls back to whatever's there" |
| 170 | `serving_bench` | TTFT/TPOT/throughput | TPOT ~8.6x TTFT — a direct, measured consequence of no KV cache in the base transformer forward pass |
| 171 | *(synthesis)* | — | Pick one number from serving_bench and explain in your own words why it comes out the way it does |

---

## Phase 10: Observability (`observability/`) — Days 172-181

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 172 | `ebpf` | Real BCC tracepoint/kprobe program | Linux-only, unrun here |
| 173 | `opentelemetry` | Hand-rolled span lifecycle + OTLP-JSON export | Parses its own exported JSON back |
| 174 | `dashboard` | Ingestion/histogram/report pipeline | Run against synthetic data shaped like serving_bench's real numbers |
| 175 | `tlc` | TLC model-checking configs | For the Raft/Collective TLA+ specs — Java-gated |
| 176 | `symbiyosys_ci` | Re-runs the real symbiyosys proofs | With SHA-256 change detection — caught a real bash-3.2 portability bug |
| 177 | `chaos` | 2/3 scenarios run for real | Raft leader-kill (183.7ms recovery), FPGA thermal event |
| 178 | `nsight_agent` | Real Anthropic SDK client, no live calls made | An explicit budget decision, not a gap |
| 179 | `kernel_variant_agent` | Same pattern as nsight_agent | — |
| 180 | `llm_autotune` | Same pattern again | Read CLAUDE.md's reasoning for deliberately not calling a real API here |
| 181 | *(synthesis)* | — | Would you have made the same call on the 3 agent steps? Why or why not |

---

## Phase 12: Machine Learning Library (`ml/`) — Days 182-200
All 18 steps actually run locally.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 182 | `decision_tree` | CART, Gini/entropy | — |
| 183 | `random_forest` | Bagging + OOB error + permutation importance | — |
| 184 | `gradient_boosting` | Newton-step GBT (Friedman 2001) | — |
| 185 | `svm` | Simplified SMO (Platt 1998) | — |
| 186 | `knn` | KD-tree + ball-tree | Branch-and-bound verified against brute force |
| 187 | `kmeans` | k-means++ | ~24x lower inertia than random init |
| 188 | `pca` | Randomized SVD | Real bug: float32 precision loss summed explained-variance-ratio above 1.0 — fixed by moving internals to double |
| 189 | `linear_models` | SGD elastic-net + L-BFGS | — |
| 190 | `openml_bench` | Real OpenML REST API fetch | Hand-written ARFF loader, no sklearn |
| 191 | `cross_method` + `decision_criteria` | Written analyses | Grounded in openml_bench's real numbers |
| 192 | `hyperparam_sensitivity` | Per-algorithm hyperparameter sweep | The step that found pca's float32 bug |
| 193 | `ensemble` | Diversity proof | Real data nuance: naive majority voting can hurt a diverse-but-unequal ensemble |
| 194 | `failure_modes` | One measured failure per algorithm | The single most "aha"-dense file in the phase — read this one closely |
| 195 | `bayesian_opt` | Exact GP regression | Cholesky-based, double-precision internals |
| 196 | `tpe` | Parzen-window density estimates | Bergstra 2011 bandwidth heuristic |
| 197 | `hyperband` | ASHA | Real bug: a 1-config rung trivially "won" its own group — fixed by requiring ≥eta results |
| 198 | `pbt` | Population-based training | Genuine warm-start via real partial_fit/set_weights, not simulated continuation |
| 199 | *(catch-up)* | Re-read anything above you rushed | — |
| 200 | *(synthesis)* | — | All 4 HPO methods independently found random search hard to beat here. Why might that be true on these datasets specifically? |

---

## Phase 13: RAG (`rag/`) — Days 201-209
All 9 steps actually run locally, zero hardware gate.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 201 | `embedding_model` | Bidirectional encoder, contrastive InfoNCE | Retrieval accuracy 0.333→1.000 after training |
| 202 | `hnsw` (+ `cosine_ann` in `ml/knn`) | HNSW vs. BallTree, cosine via L2-normalize | Honest finding: exact BallTree beat approximate HNSW at this corpus's scale |
| 203 | `indexing_pipeline` (+ `corpus`) | Chunking + embed + index | 100% top-1 retrieval end to end |
| 204 | `rag_generation` | Prompt construction + generation | Composes `inference_serving`'s CPU backend rather than reimplementing decode |
| 205 | `recall_eval` | recall@1/3/5 exact vs approximate | Distractor docs exist specifically to make exact/approximate diverge at all |
| 206 | `generation_quality` | QA trained to answer-or-abstain | 0.000→1.000 accuracy with vs. without retrieval |
| 207 | `approx_retrieval_study` | Composition of recall_eval + generation_quality | Recall drop costs exactly as much generation accuracy — no more, no less |
| 208 | `serving_integration` | `route_rag()` wired into ServingRouter | Without modifying ServingRouter itself — genuine delegation, not a parallel path |
| 209 | *(synthesis)* | `rag/DESIGN.md` | The tightest single argument-chain in the repo — read it in one sitting |

---

## Phase 14: Adversarial Robustness (`adversarial/`) — Days 210-218
All 8 steps (incl. the stretch goal) actually run locally.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 210 | `input_gradients` | x.grad() via the existing autograd tape | Real fix: zero weight grads before each call so PGD's iterations don't silently accumulate stale ones |
| 211 | `fgsm` | Goodfellow 2014 | Verified the L∞ bound is exact |
| 212 | `pgd` | Madry 2017 | Verified PGD(1 step)=FGSM exactly — a checkable structural claim, not just argued |
| 213 | `vulnerability_measurement` | Accuracy collapse curve | 1.000→0.011 as epsilon grows |
| 214 | `adversarial_training` | Madry's min-max | Robust accuracy 0.722→0.844 |
| 215 | `robustness_tradeoff` | Tsipras 2018's cost-of-robustness | Measured directly across an epsilon sweep |
| 216 | `transferability` | Cross-model attack transfer | 37.8% vs 2.2% random-noise baseline — the gap proving genuine transfer |
| 217 | `randomized_smoothing` | Cohen 2019 certified defense | The stretch goal, actually reached, one disclosed simplification |
| 218 | *(synthesis)* | — | pgd=fgsm-at-1-step is a nice self-check pattern. Where else does a phase verify itself this way? (hint: dpo/reward_model share a loss function) |

---

## Phase 15: NPU Backend (`npu_engine/`) — Days 219-223

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 219 | `quant_export` | Reuses inference_serving's GptqQuantizer unchanged | Composition over reimplementation, at bits=8 instead of 4 |
| 220 | `cost_model` | NPU vs GPU vs CPU efficiency | Two-sided finding: NPU wins power efficiency, loses on large-workload latency |
| 221 | `op_coverage` | Walks all 18 dialect ops | Found gather/scatter aren't placed for ANY device yet — a real Phase 4 gap surfaced as a side effect |
| 222 | `thermal` | Mirrors fpga_engine/thermal_router's split | Discloses a real gap: no stable per-ANE thermal sensor on Apple Silicon |
| 223 | *(synthesis)* | — | Note every other cross-phase dependency you've spotted so far (triton_kernels←kernels, mxu_opt↔systolic_sweep, op_coverage→placement) |

---

## Phase 16: Containers & Orchestration — Days 224-228
Cross-cutting; real, unrun (no Docker/kubectl/cluster locally).

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 224 | `Dockerfile` + `docker/gpu` | Multi-stage build + GPU passthrough | — |
| 225 | `k8s/serving` | Latency-driven HPA | Deliberately not raw CPU% — the design choice PLAN.md asked for |
| 226 | `k8s/training` | StatefulSet, `podManagementPolicy: Parallel` | Gang-scheduled training |
| 227 | `distributed_training/serving_daemon` + `training_worker` | The two real components these manifests forced into existence | Writing the spec exposed missing pieces that then had to become real |
| 228 | *(synthesis)* | — | What's the analogous "the spec exposed a missing piece" moment in an earlier phase? |

---

## Phase 17: Analog & Unconventional Compute (`analog_engine/`) — Days 229-237
Zero hardware gate — fully local, fully run. **Most directly relevant
phase to real physical-hardware roles.**

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 229 | `device_model` | RRAM noise/drift/endurance model | Real bug: endurance check re-rolled a Bernoulli every write, compounding wrongly — fixed via a fixed per-cell threshold |
| 230 | `crossbar_mac` | Signed-weight differential-pair analog MAC | Precision drives accuracy far more than crossbar size |
| 231 | `nvm_comparison` | RRAM vs PCM vs STT-MRAM vs SRAM-CIM | RRAM wins on having no hard disqualifier, not on any single axis |
| 232 | `energy_model` | Analog vs digital MAC energy | ADC overhead dominates by 121-361x — a real debate in the CIM literature |
| 233 | `dataflow_model` | From-scratch Timeloop/Accelergy-style tool | Confirms Row-Stationary gets WS's weight-movement AND IS's input-movement at once |
| 234 | `systolic_sweep` | PE-array size sweep on real GEMM shapes | Independently reproduces tpu_engine/mxu_opt's utilization-cliff finding |
| 235 | `codesign_case_study` | GPTQ re-derived under device noise | A real non-monotonic result the test caught, leading to a defensible (not overclaimed) assertion |
| 236 | `circuit_transient` | RC step-response, tau vs crossbar size | Tau scales quadratically — a third independent reason bigger crossbars aren't free |
| 237 | *(synthesis)* | Pick `crossbar_mac` or `device_model` | Be ready to explain it out loud, no notes — this is the "go deep on one thing" piece worth doing for real |

---

## Phase 18: SciML / Dynamical Systems (`sciml/`) — Days 238-248
Fully CPU-portable, fully run.

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 238 | `ode_solver` | Euler/RK4/backward Euler | RK4 converges at 16.11x per dt-halving vs. theoretical 16x |
| 239 | `stiffness` | Forward Euler's stability boundary | Van der Pol stiffness ratio grows to ~2498 |
| 240 | `sde_solver` | Euler-Maruyama/Milstein | Milstein reduces exactly to EM for additive noise |
| 241 | `neural_ode` | Adjoint-method gradients | Real sign bug caught by finite-difference check — every param's relative error was exactly 2.0 before the fix |
| 242 | `deq` | Implicit-function-theorem backprop | No bug this time — gradient check passed first try |
| 243 | `ssm_layer` | Generic linear SSM vs self-attention | Attention wins a long-range copy task — S4's real edge is HiPPO init, not just linear recurrence |
| 244 | `diffusion` | Real trained DDPM, toy 2-cluster data | 200 samples split 94/106 — not mode-collapsed |
| 245 | `ebm` | Contrastive divergence + Langevin MCMC | Directly compared to diffusion in the same binary — diffusion clearly wins |
| 246 | `mup_scaling` | muP's LR-scaling claim | Best LR lands at exactly 1.0 across widths 4/8/16, no tuning |
| 247 | `noise_aware_training` | Training through the analog device-noise model | Real bug: naive finite-differencing through quantization+noise diverged — why real QAT needs a Straight-Through Estimator |
| 248 | *(synthesis)* | — | neural_ode's exactly-2.0 error and analog's exactly-4.0%-stuck number are both "the number told us the bug." Find a third one somewhere else in the repo |

---

## Phase 19: Framework-Native Training (`framework_native/`) — Days 249-255
Actually run in a real `.venv` (torch/jax/ray installed).

| Day | Dir | Read/do | Why it matters |
|---|---|---|---|
| 249 | `pytorch_transformer` | Real torch.nn.Module port of transformer/ | Caught a real numpy 2.x/torch ABI break before any model code even ran |
| 250 | `torch_compile_bench` | `torch.compile()` on this platform | Verified a genuine hard gate — Dynamo doesn't support Python 3.12+ here |
| 251 | `ddp_gloo` | Real DDP over 4 OS processes | Matches the hand-rolled data_parallel baseline to 0.000000 max difference |
| 252 | `fsdp_vs_zero` | Real FSDP mapped to ZeRO-3 | 3 real bugs found (CUDA-fallback crash, rank-gated hang, sharded-storage crash) before landing on: each rank holds exactly 25.0% of params |
| 253 | `jax_transformer` | Third independent implementation | Same architecture/task as C++ and PyTorch — pmap-checked across 4 simulated devices |
| 254 | `production_framework` | DeepSpeed hit 2 real platform walls | Ray Train used as the real working fallback instead |
| 255 | *(synthesis)* | — | Three independent implementations converged on the same architecture. What does that prove, and what does it NOT prove? |

---

## Day 256 — Capstone
Re-read CLAUDE.md's "Where we are" section top to bottom, now that you've
seen the code behind almost every line of it. Write your own 3-sentence
summary of the whole project, in your own words, as if explaining it to
an interviewer.
