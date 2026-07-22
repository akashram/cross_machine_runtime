# Cross-Machine Runtime — Phased Implementation Plan

## Guiding principles
- Depth-first within each phase. Nothing moves forward until the current phase is done to embarrassing depth.
- Phases are ordered by hard dependency. Later phases build on earlier ones.
- Observability (benchmarking harness, sanitizers, perf counters) is set up in Phase 1 and applied to every subsequent phase — not added at the end.
- "Done" means: benchmarks with confidence intervals, design doc explaining WHY each decision was made, comparison against the alternative you didn't choose, hardware counter analysis where applicable, and code that passes ASan/TSan/UBSan.

## Execution order (updated 2026-07-19)

**Write all code for every phase before spending a dollar on cloud hardware.**
Phases below run in this same numbering order (1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
→ 9 → 10 → 12) for both implementation and hardware validation — there is no
separate reordering for the code-writing pass. A phase is "code-complete"
once every step in its build order has real logic (not a stub) and compiles
wherever it can without the target hardware; benchmark numbers stay TODO
until that phase's hardware validation pass.

Only after every phase below is code-complete does cloud hardware get
provisioned, one phase at a time, in the same order, to run and tune what
was already written. See CLAUDE.md for current phase status and the
hardware-per-phase table.

---

## Phase 1: Foundation
**Estimated duration: 3–4 months**

Everything else in this project depends on what you build here. Do not rush this phase.

### Cloud infrastructure needed
None. This phase runs entirely on your Mac.

### What to learn first
- C++20: coroutines (promise_type, awaitable, awaiter, co_await/co_yield/co_return), concepts, ranges
- C++23: std::expected, std::mdspan, deducing this, constexpr additions
- x86 memory model (TSO — Total Store Order): why x86 is strongly ordered and what that means for lock-free code
- C++ memory model: std::memory_order_relaxed/acquire/release/acq_rel/seq_cst — how they map to x86 instructions
- Cache coherence: MESI state machine, why false sharing is a coherence artifact, not just a performance quirk

### Build order
1. **CMake project skeleton** — modern target-based CMake (3.25+), vcpkg for dependencies, sanitizer presets (ASan/TSan/UBSan as CMake profiles), CI pipeline (GitHub Actions), static analysis (clang-tidy)
2. **Statistical benchmarking harness** — TSC-based (RDTSC/RDTSCP) nanosecond timing, warmup logic, confidence interval computation, outlier detection, CSV/JSON output, performance regression CI hook
3. **SPSC ring buffer** — single-producer single-consumer, cache-line aligned, power-of-two size, acquire/release ordering, measured throughput and latency under contention
4. **MPMC ring buffer** — multi-producer multi-consumer, compare-and-swap based, documented ABA analysis
5. **ABA problem** — demonstrate the bug with a concrete example, then fix with tagged pointers, document why each solution works
6. **Hazard pointers** — implement from scratch, document the algorithm, compare retire latency vs. tagged pointer approach
7. **Epoch-based reclamation** — implement alongside hazard pointers, document when to choose each
8. **RCU (Read-Copy-Update)** — userspace RCU, read-side critical sections, grace period detection
9. **Lock-free freelist** — using hazard pointers for safe reclamation
10. **Lock-free queue (Michael-Scott)** — two-lock-free queue, validated with TSan
11. **Chase-Lev work-stealing deque** — the foundational scheduler data structure, proven correct in the literature, implement and validate
12. **Work-stealing thread pool** — built on Chase-Lev deque, task graph with dependencies, C++20 coroutine tasks
13. **Coroutine execution engine** — async task type using C++20 coroutines, awaitable event, awaitable mutex, coroutine-aware thread pool integration
14. **Arena allocator** — per-thread arenas, lock-free freelist per size class, alignment-aware, hugepage support (mmap with MAP_HUGETLB), benchmark vs. system malloc
15. **NUMA-aware allocator** — per-NUMA-node arenas, libnuma integration, thread-to-NUMA binding, measured cross-NUMA penalty
16. **Unified tensor handle (v1)** — CPU-only at this stage: device tag, data pointer, shape, strides, dtype, ownership model, ref-counted, no virtual dispatch (CRTP or std::variant)
17. **Property-based testing setup** — rapidcheck or Catch2 with generators, write invariant tests for all lock-free data structures (linearizability, no lost updates, no ABA)
18. **x86 hardware counter infrastructure** — perf_event_open() wrapper, measure cache misses / branch mispredictions / IPC for every benchmark from this point forward

### Deliverables
- `/foundation/` directory with all of the above
- Design doc: "Memory Ordering in the Runtime" — documents every std::memory_order choice and why
- Design doc: "Lock-Free Memory Reclamation" — hazard pointers vs. epoch-based vs. tagged pointers, when to use each
- Benchmark report: all data structures at p50/p99/p999 latency and throughput under varying contention levels

### Definition of done
TSan finds zero races. Every data structure has property-based tests that verify linearizability. Benchmarks include hardware counter data (IPC, cache miss rate). Design docs are written to Google internal doc standard.

---

## Phase 2: CPU Backend
**Estimated duration: 2–3 months**

### Cloud infrastructure needed
None. Mac only.

### What to learn first
- x86 microarchitecture: pipeline stages, ROB (reorder buffer), store buffer, load buffer, execution ports (use Agner Fog's instruction tables)
- AVX-512 intrinsics: `_mm512_*` family, gather/scatter, mask registers, fused multiply-add
- Branch predictor internals: bimodal predictor, TAGE predictor, how to write predictor-friendly code
- Cache hierarchy: L1/L2/L3 sizes on your target instance type, associativity, TLB structure
- `perf` tool: `perf stat`, `perf record`, `perf annotate`, `perf mem`

### Build order
1. **CPU affinity + thread pinning** — `pthread_setaffinity_np`, `sched_setaffinity`, verify pinning works, measure jitter with and without
2. **Hugepage allocator** — explicit 2MB pages via `mmap(MAP_HUGETLB)`, measure TLB miss reduction vs. 4KB pages with `perf stat -e dTLB-misses`
3. **OS-level tuning scripts** — `isolcpus`, `nohz_full`, IRQ affinity (`/proc/irq/*/smp_affinity`), C-state disabling (`/sys/devices/system/cpu/cpu*/cpuidle`), CPU governor (`performance`). Packaged as reproducible shell scripts with before/after latency measurements.
4. **Non-temporal store primitives** — `_mm_stream_*` wrappers, benchmark write-only paths vs. regular stores, document when to use (write-only, large sequential writes)
5. **Prefetch primitives** — `__builtin_prefetch` wrappers with T0/T1/T2/NTA hints, measure prefetch distance vs. cache miss rate
6. **Branchless primitives** — conditional move patterns (`cmov`), branchless min/max/abs/clamp, verify with `perf stat -e branch-misses`
7. **AVX-512 kernel library** — vectorized dot product, matrix-vector multiply, element-wise ops (add/mul/relu/sigmoid), INT8 ops. Each kernel: measure throughput vs. latency, compare against scalar baseline and compiler auto-vectorized version.
8. **Cache-aware tiling** — blocked matrix multiply with tile sizes tuned to L1/L2 cache, measure cache miss rates at each tile size, document optimal tiling strategy
9. **CPU inference engine** — run a small MLP (matching the model from the router design) through the AVX-512 kernel library, preallocated buffers, no heap on hot path, verified with Valgrind massif
10. **Roofline model (CPU)** — measure peak FLOPS (via micro-benchmark), peak memory bandwidth (STREAM benchmark), plot achieved FLOPS vs. bandwidth for each kernel, classify compute-bound vs. bandwidth-bound
11. **Hardware perf counter deep dive** — for every kernel: IPC, L1/L2/L3 miss rates, branch misprediction rate, TLB miss rate. Document and explain each anomaly.
12. **Profile-guided optimization (PGO)** — instrument build, collect profiles from representative workload, recompile with `-fprofile-use`, measure improvement
13. **Busy-poll vs OS-wait comparison** — implement both approaches for a producer/consumer scenario, measure latency at p50/p99/p999, document the crossover point

### Deliverables
- `/cpu_engine/` with all kernels, benchmarks, and OS tuning scripts
- Design doc: "CPU Backend Architecture" — every optimization decision with measured justification
- Roofline charts for all major kernels
- Hardware counter analysis report

### Definition of done
Every kernel is within 80% of the roofline ceiling. Latency measurements include hardware counter data. PGO improvement is measured and documented. OS tuning scripts are reproducible and version-controlled.

---

## Phase 3: GPU Backend (Single Node)
**Estimated duration: 4–5 months**

### Cloud infrastructure needed
AWS GPU instance — start with `g4dn.xlarge` (T4, cheapest) for learning, move to `p3.2xlarge` (V100) for serious work, `p4d.24xlarge` (8x A100 NVLink) for multi-GPU and NVLink work. Hopper (H100) via `p5.48xlarge` for TMA/WGMMA work.

### What to learn first
- CUDA programming model: grids, blocks, threads, warps, SMs
- CUDA memory hierarchy: registers, shared memory, L1/L2, global, constant, texture
- Warp execution: SIMT, divergence, `__syncwarp`, `__syncthreads`
- CUDA streams and events: asynchronous execution, inter-stream synchronization
- Nsight Compute: roofline view, memory chart, warp state statistics, source correlation

### Build order
1. **CUDA project integration** — CMake CUDA support, device detection, compute capability targeting, nvcc flags
2. **GPU memory management** — cudaMalloc/cudaFree wrappers, pinned host memory (cudaMallocHost), unified tensor handle extended for GPU device
3. **Stream manager** — pool of CUDA streams, stream assignment per operation, event-based synchronization, compute/transfer overlap
4. **Warp-level primitives library** — `__shfl_sync` for warp broadcast/reduce, `__ballot_sync` for predicate aggregation, `__reduce_sync` variants, warp scan. Each with Nsight validation.
5. **Shared memory primitives** — bank conflict-free reduction, prefix sum, transpose with padding, documented bank conflict analysis for each
6. **Memory coalescing validator** — Nsight Compute memory access pattern analysis integrated into CI, flag any kernel with < 90% coalesced access
7. **Occupancy tuner** — `cudaOccupancyMaxActiveBlocksPerMultiprocessor` integration, measure occupancy vs. register count vs. shared memory, document tradeoffs per kernel
8. **Element-wise GPU kernels** — add, mul, relu, gelu, softmax — optimized for coalescing and occupancy, roofline analysis
9. **GEMM kernel** — start with naive, then shared-memory tiled, then tensor core via WMMA API, then cuBLAS comparison. Document each step's performance and why.
10. **PTX/SASS inspection workflow** — `cuobjdump --dump-sass`, identify inefficiencies (unnecessary memory transactions, suboptimal instruction scheduling), document findings
11. **Flash Attention forward kernel** — tiled SRAM implementation (Dao et al. algorithm), online softmax, causal masking. Benchmark vs. cuDNN attention and naive attention.
12. **Flash Attention backward kernel** — recomputation of attention weights from output, gradient through softmax. Validate numerically against autograd reference.
13. **CUDA Graphs** — capture a full forward pass as a CUDA graph, replay, measure CPU overhead reduction vs. eager execution
14. **GPUDirect P2P** — enable peer access between GPUs, direct cudaMemcpyPeerAsync, measure bandwidth vs. host-staged transfer
15. **Mixed precision** — BF16 forward pass, FP32 master weights, loss scaling with dynamic scale factor, validate convergence on toy model
16. **FP8 (Hopper)** — `__nv_fp8_e4m3` / `__nv_fp8_e5m2`, Hopper FP8 Tensor Core via WGMMA, measure vs. BF16 baseline
17. **Tensor Core alignment analysis** — demonstrate perf cliff when dimensions aren't multiples of 8/16/32, document alignment requirements per dtype
18. **Hopper TMA** — async bulk tensor copy with `cuda::memcpy_async` and TMA descriptors, measure vs. standard cudaMemcpyAsync
19. **Hopper WGMMA** — warpgroup-level matrix multiply, compare against WMMA API, measure throughput improvement
20. **2:4 structured sparsity** — prune a weight matrix to 2:4 pattern, use `cusparseLtMatmul` for sparse matmul, measure 2x throughput, document accuracy impact
21. **Roofline model (GPU)** — peak FLOPS via cublas benchmark, peak HBM bandwidth via bandwidth test, plot achieved vs. ceiling for every kernel
22. **CUDA MPS setup** — configure MPS server, measure context-switching overhead reduction for multi-process GPU sharing
23. **NVML power monitoring** — power draw per kernel, thermal throttling detection, integrate into benchmark harness
24. **Nsight integration in CI** — `ncu --set full` profiling as part of benchmark pipeline, parse and store metrics per commit

### Deliverables
- `/gpu_engine/` with all kernels
- Roofline charts per kernel with hardware counter breakdown
- Flash Attention benchmark vs. cuDNN (should beat it)
- PTX/SASS analysis writeup for 2-3 key kernels
- Design doc: "GPU Backend Architecture"

### Definition of done
Every kernel has Nsight analysis showing occupancy, memory efficiency, and compute utilization. Flash Attention beats cuDNN. Roofline analysis shows all kernels within 85% of ceiling. Mixed precision training loop produces correct gradients (validated numerically).

---

## Phase 4: Compiler / IR (MLIR)
**Estimated duration: 5–6 months**

### Cloud infrastructure needed
None new. Mac + existing GPU instance for validation.

### What to learn first
- MLIR architecture: dialects, ops, types, attributes, regions, blocks, values
- Existing MLIR dialects: Affine, Linalg, Arith, Func, SCF, Tensor, Bufferization
- MLIR Affine dialect: affine maps, affine for/if, polyhedral analysis
- LLVM TableGen: how ops and types are defined
- MLIR pass infrastructure: pass manager, pattern rewriting (RewritePatternSet, ConversionTarget)
- Read: "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation" (CGO 2021)

### Build order
1. **MLIR build setup** — build LLVM/MLIR from source, CMake integration, understand the build system
2. **Runtime dialect design** — define your tensor op IR: ops for matmul, conv, elementwise, reduce, scatter/gather. Types for tensors with device placement annotation. Attributes for precision, layout.
3. **Dialect registration + parsing** — TableGen op definitions, custom assembly format, round-trip test (parse → print → parse)
4. **Shape inference pass** — propagate shapes through the op graph, handle dynamic shapes via symbolic dimensions
5. **Operator fusion pass** — pattern-match fusable op sequences (e.g., matmul + bias + relu → fused op), measure memory traffic reduction
6. **Affine dialect lowering** — lower loop nests to Affine dialect for polyhedral analysis, apply tiling and interchange transformations
7. **Memory planning pass** — liveness analysis on tensor values, buffer aliasing (reuse buffers with non-overlapping lifetimes), peak memory minimization. Validate against XLA's memory planning behavior.
8. **Rematerialization pass** — identify tensors that are cheaper to recompute than to store (activation checkpointing), insert recompute ops, measure memory vs. compute tradeoff
9. **Device placement pass** — cost model (FLOP count, memory size, transfer cost), assign each op to CPU/GPU/FPGA/TPU, minimize total cost including data movement
10. **Auto-sharding pass** — GSPMD-style: annotate tensors with sharding specs, propagate sharding through ops, insert communication ops (all-gather, reduce-scatter) where sharding changes
11. **Kernel specialization** — lower dialect ops to device-specific implementations (select AVX-512 kernel vs. CUDA kernel vs. HLS kernel based on placement)
12. **AOT compilation pipeline** — end-to-end: parse IR → optimize → lower → codegen → binary. Time compilation, measure generated code quality vs. hand-written.
13. **Cost model** — calibrated model for each device: FLOPS/sec, memory bandwidth, launch overhead, transfer bandwidth. Used by placement and sharding passes.
14. **libFuzzer integration** — fuzz the IR parser, fuzz the pass pipeline with random-but-valid IR, document bugs found
15. **LLVM upstream** — while working on MLIR, note any bugs or improvements. File issues or patches upstream.

### Deliverables
- `/compiler/` with dialect, passes, and AOT pipeline
- End-to-end demo: write a model in the IR, compile to CPU+GPU, run faster than PyTorch eager
- Design doc: "IR Design and Compilation Pipeline"
- Design doc: "Memory Planning Algorithm"
- Fuzzer corpus and any upstream LLVM contributions

### Definition of done
Full compilation pipeline works end-to-end. Memory planning measurably reduces peak memory vs. naive allocation. Fusion pass reduces memory bandwidth. AOT-compiled model runs at parity or better than hand-written kernel.

---

## Phase 5: Distributed Layer + Networking
**Estimated duration: 5–6 months**

### Cloud infrastructure needed
Multi-node AWS setup: at least 2x `p4d.24xlarge` instances (EFA-enabled) in a placement group (for low-latency EFA). VPC setup, security groups, placement group configuration.

### What to learn first
- RDMA concepts: memory registration, queue pairs (QP), completion queues (CQ), work requests (WR), scatter/gather elements (SGE)
- libfabric / OFI API: fi_domain, fi_endpoint, fi_mr_reg, fi_send/fi_recv, fi_read/fi_write
- EFA specifics: SRD transport, AWS EFA installer, EFA-enabled instance setup
- gRPC: service definition, streaming RPCs, channel credentials, interceptors
- PTP: IEEE 1588, `linuxptp` stack, `ptp4l` and `phc2sys` daemons
- Raft: Ongaro's thesis ("In Search of an Understandable Consensus Algorithm"), TLA+ spec

### Build order
1. **EFA setup and validation** — AWS EFA installer, `fi_info`, `fi_pingpong` benchmark, establish baseline latency and bandwidth numbers
2. **libfabric RDMA transport (v1)** — memory registration, two-sided send/recv, measure latency vs. TCP baseline
3. **One-sided RDMA operations** — fi_read/fi_write (RDMA read/write without remote CPU involvement), compare latency vs. two-sided, document when each is appropriate
4. **EFA SRD transport** — understand SRD vs. RC (reliable connected), when SRD's scalable datagram model wins
5. **PTP clock synchronization** — set up `ptp4l` on EC2 instances, measure synchronization accuracy, compare against NTP baseline. Integrate timestamps into benchmark harness.
6. **gRPC + protobuf control plane** — service definitions for scheduler RPC, node registration, health checks, task dispatch
7. **Flatbuffers data plane** — tensor descriptor serialization without copying, benchmark vs. protobuf for hot-path messages
8. **AF_XDP kernel bypass** — XDP socket setup, UMEM region, TX/RX rings, measure latency vs. standard socket
9. **Userspace networking stack** — built on AF_XDP or DPDK, send/recv pipeline with measured latency
10. **NIC hardware deep dive** — document TX/RX descriptor ring structure, RSS configuration, PFC + ECN setup for lossless fabric (DCQCN), hardware timestamping configuration
11. **Ring all-reduce** — implement from scratch, measure bandwidth efficiency vs. theoretical, compare against NCCL baseline
12. **Recursive halving-doubling all-reduce** — implement, compare against ring for small vs. large message sizes, document crossover point
13. **Tree all-reduce** — implement, measure, three-way comparison with topology analysis
14. **Broadcast, reduce-scatter, all-gather** — complete collective library
15. **NCCL integration + tuning** — `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_BUFFSIZE` tuning for EFA topology, measure collective throughput before/after tuning
16. **Topology-aware scheduler** — PCIe tree discovery, NVLink topology (nvml), EFA bandwidth map, FPGA attachment paths. Scheduler uses this map for placement decisions.
17. **Vector clocks** — implement Lamport timestamps and vector clocks, integrate into event logging
18. **Chandy-Lamport snapshots** — distributed global state capture without stopping execution, validate consistency of captured snapshots
19. **Raft consensus** — leader election, log replication, membership changes. Written from scratch.
20. **TLA+ spec for Raft** — model the implementation in TLA+, run TLC model checker, verify liveness and safety properties
21. **Backpressure + load shedding** — token bucket rate limiting, explicit backpressure signals between nodes, graceful degradation under overload
22. **Hedged requests** — duplicate slow requests to a second backend after p95 latency threshold, measure tail latency improvement
23. **Multi-tenancy** — resource quota enforcement, priority preemption, fair scheduling with measured isolation
24. **Chaos engineering harness** — `tc netem` for latency injection / packet loss / reordering, node kill scripts, GPU OOM injection. Automated recovery validation.
25. **TLA+ for collective protocol** — formally verify that the all-reduce protocol is deadlock-free and produces consistent results

### Deliverables
- `/networking/` and `/distributed/` directories
- Collective benchmark vs. NCCL (ring/tree/halving-doubling comparison)
- PTP synchronization accuracy report
- TLA+ specs for Raft and collective protocol
- Design doc: "Distributed Transport Architecture"
- Chaos test suite results

### Definition of done
All-reduce within 10% of NCCL throughput. PTP synchronization < 1µs across nodes. Raft passes TLC model checker with no violations. Chaos tests show graceful recovery from all injected failure modes.

---

## Phase 6: Distributed GPU Training
**Estimated duration: 3–4 months**

### Cloud infrastructure needed
`p4d.24xlarge` (8x A100 with NVLink + EFA) — at minimum 2 nodes for multi-node training.

### What to learn first
- Megatron-LM paper: tensor parallelism and pipeline parallelism designs
- ZeRO paper (Rajbhandari et al.): ZeRO-1/2/3 memory reduction analysis
- 1F1B schedule: "PipeDream: Generalized Pipeline Parallelism for DNN Training"
- GPipe bubble fraction analysis
- MoE routing: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- InstructGPT (Ouyang et al. 2022): how RLHF with PPO aligns a pretrained LLM — read before implementing steps 22–24
- DPO (Rafailov et al. 2023): Direct Preference Optimization as a simpler PPO alternative — understand the reparameterization that eliminates the reward model

### Minimal Transformer (added 2026-07-21, not in the original 12-phase scope)
Steps 22–25 (SFT, reward model, PPO, DPO) need an actual model to train,
and nothing earlier in this plan builds one — Phase 9 (Inference Serving)
assumes an existing transformer to serve, it does not build one either.
`/transformer/` fills that gap: a minimal decoder-only transformer (token
+ positional embedding, N pre-LN blocks with causal multi-head
self-attention and an MLP, final LayerNorm, output projection) plus a
character-level tokenizer, both real and complete — real causal masking,
real backprop through the whole stack (hand-derived at the Matrix level,
reusing `tensor_parallel_attn`'s attention primitive and
`seq_parallel`'s LayerNorm), gradient-checked, and validated by an actual
training run that greedy-generates its training corpus back exactly. No
BPE tokenizer pipeline (character-level only) and no batching (one
sequence per forward call) — stated scope choices, see
`transformer/README.md`. This is a standalone component, not scoped to
steps 22–25 alone — Phase 9 (`inference_serving/`) can use it too if that
phase is ever implemented.

### Build order
1. **Distributed data loading** — WebDataset format, multi-worker DataLoader, dataset sharding across ranks, prefetch queue, measure GPU utilization with and without pipeline. Target: data loading never the bottleneck.
2. **GPUDirect Storage** — direct NVMe → GPU checkpoint loading, measure load time vs. CPU-staged loading
3. **Data parallel training (baseline)** — manual gradient all-reduce after backward pass, validate loss curve matches single-GPU baseline
4. **Gradient accumulation** — micro-batch accumulation with correct gradient scaling, interaction with loss scaling
5. **Gradient clipping** — distributed gradient norm all-reduce, clip by global norm
6. **Autograd engine** — reverse-mode tape-based autograd for the ops in your dialect. Or: explicit interface to PyTorch autograd with measured overhead. Document the decision.
7. **ZeRO-1** — shard optimizer states (momentum, variance) across data parallel ranks. Measure memory reduction, validate correctness.
8. **ZeRO-2** — add gradient sharding. Measure memory reduction.
9. **ZeRO-3** — add parameter sharding. Measure memory reduction vs. ZeRO-1/2, document communication overhead.
10. **ZeRO-Infinity** — CPU RAM offload, NVMe offload, measure throughput degradation vs. memory capacity gain
11. **Column/row-parallel linear layers** — tensor parallel linear: split weight matrix column-wise for forward, all-reduce output; row-wise for second linear, reduce-scatter. Validate numerically.
12. **Tensor-parallel attention** — split attention heads across tensor parallel ranks
13. **Sequence parallelism** — shard sequence dimension, interleave with tensor parallelism
14. **1F1B pipeline schedule** — interleaved schedule, microbatch assignment, bubble fraction measurement vs. GPipe
15. **3D parallelism** — combine data + tensor + pipeline. Verify all combinations work correctly.
16. **MoE / Expert parallelism** — top-k routing, expert dispatch (all-to-all), expert forward pass on assigned tokens, combine results. Validate on small MoE model.
17. **Checkpoint sharding** — sharded checkpoint format, async write overlapped with training, fast restore. Measure checkpoint time for a 7B parameter model equivalent.
18. **Compute/communication overlap** — double-buffer backward pass with all-reduce: while computing layer N gradients, communicate layer N+1 gradients. Measure throughput improvement.
19. **Distributed batch normalization** — SyncBatchNorm with all-reduce of mean/variance
20. **Full training loop** — forward → backward → grad sync → ZeRO optimizer step → checkpoint. Latency breakdown per phase, identify bottleneck.
21. **2:4 sparsity in training** — structured pruning to 2:4 pattern during training, validate convergence, measure throughput
22. **Supervised fine-tuning (SFT)** — instruction-response dataset, per-rank data sharding, loss masking on prompt tokens, validate perplexity improvement over base model checkpoint
23. **Reward model training** — preference pairs (chosen/rejected), Bradley-Terry objective, measure ranking accuracy on held-out preference set, verify reward signal is meaningful before proceeding
24. **PPO-based RLHF** — policy (SFT init) + critic (reward model) + KL penalty against frozen reference model, clip ratio, measure reward vs. KL divergence tradeoff across training, monitor for reward hacking
25. **DPO (Direct Preference Optimization)** — offline alternative to PPO: optimize policy directly on preference pairs without a reward model, compare convergence speed and final reward vs. PPO baseline on identical preference data, document when to prefer each

### Deliverables
- `/distributed_training/` directory
- Scaling efficiency chart: throughput vs. number of GPUs (data parallel, tensor parallel, pipeline parallel, 3D)
- ZeRO memory analysis: memory per rank at each stage
- MoE routing analysis: expert utilization distribution
- RLHF vs. DPO comparison: reward, KL divergence, training stability, wall-clock time
- Design doc: "Distributed Training Architecture"

### Definition of done
Linear scaling efficiency > 85% from 1→8 GPUs (data parallel). ZeRO-3 enables training a model 8x larger than fits on a single GPU. 1F1B pipeline bubble fraction < 5% with sufficient microbatches. DPO training loop produces a measurably higher-reward policy than SFT baseline on held-out prompts.

---

## Phase 7: FPGA Backend
**Estimated duration: 5–6 months**

### Cloud infrastructure needed
AWS F1 instance (Xilinx UltraScale+ VU9P, 4x DDR4 banks, ~$0.50/hr spot). Development AMI: Xilinx Vitis/Vivado on Amazon Linux 2. TCL-driven headless workflow — no GUI required. X11 forwarding for occasional floorplanning/ILA sessions.

### What to learn first
- Vitis HLS: #pragma HLS PIPELINE, UNROLL, DATAFLOW, INTERFACE, BIND_OP
- AXI protocols: AXI4 (memory-mapped), AXI4-Lite (control registers), AXI4-Stream (data flow)
- FPGA fabric: LUT, DSP48E2, BRAM, URAM, CLB, routing resources
- Vivado timing analysis: setup/hold slack, critical path, WNS/TNS
- DDR4 on UltraScale+ VU9P: 4 banks, ~77 GB/s total bandwidth, AXI DDR4 IP
- Vitis platform: shell, kernel, xclbin, xrt (Xilinx Runtime)

### Build order
1. **AWS F1 setup** — launch F1 instance with Vitis AMI, validate FPGA is accessible (`xbutil examine`), run Vitis Hello World
2. **TCL build pipeline** — fully scriptable synthesis + implementation + bitstream generation, no GUI required, integrated into CI
3. **Vivado power report CI** — every bitstream automatically generates a power report, stored as CI artifact
4. **AXI4-Stream interface** — simple passthrough kernel, validate AXI stream protocol with ILA
5. **First HLS kernel: vector dot product** — naive implementation, measure II, then pipeline to II=1, verify with co-simulation
6. **Loop optimization deep dive** — `#pragma HLS UNROLL`, `#pragma HLS PIPELINE`, `#pragma HLS DATAFLOW`. For each: measure II, resource usage, Fmax. Document the tradeoffs.
7. **DSP vs LUT tradeoff analysis** — implement same arithmetic in DSP48E2 vs LUTs, measure resources, latency, power. Document when to use each.
8. **Fixed-point arithmetic** — `ap_fixed<W,I>` types, precision/resource/latency tradeoff analysis for a matrix multiply kernel
9. **BRAM vs URAM access patterns** — implement weight storage in BRAM (true dual-port) vs. URAM (single-port), measure throughput for different access patterns
10. **DDR4 integration** — connect kernel to DDR4 banks via AXI DDR4 IP, measure bandwidth per bank, multi-bank access strategy for max throughput
11. **DMA orchestration** — host → FPGA → host data flow, measure PCIe latency components (submission, transfer, completion), interrupt vs. polling for completion
12. **PCIe latency decomposition** — measure each component separately: BAR write (doorbell), DMA descriptor processing, data transfer, interrupt/poll overhead
13. **Ping-pong double buffering** — overlap computation with DMA transfer, measure throughput improvement
14. **ML inference kernel** — implement a small MLP in HLS: fully pipelined, INT8 quantized, II=1, compare latency vs. CPU and GPU baselines
15. **Timing closure** — analyze critical paths in Vivado timing report, identify top-10 critical paths, apply retiming/placement constraints to close timing. Document each fix.
16. **SLR partitioning** — assign logic to specific SLRs, measure SLR crossing penalty, document design rules for multi-SLR placement
17. **Clock gating** — implement conditional clock enables in HLS, measure dynamic power reduction with Vivado power analysis
18. **XADC monitoring** — read die temperature and voltage rails programmatically from host, integrate into thermal-aware router
19. **ILA debug session** — insert ILA probes on AXI stream interface, capture a real bug during development (or demonstrate probing a protocol)
20. **Cocotb testbenches** — Python-based RTL simulation tests for AXI stream interface and DMA controller
21. **SymbiYosys formal verification** — prove AXI stream interface never deadlocks (valid/ready handshake always resolves), prove DMA controller never issues overlapping transactions
22. **Partial reconfiguration** — define reconfigurable partition, implement two kernels as partial bitstreams, hot-swap at runtime, measure reconfiguration time
23. **FPGA network stack** — OpenNIC shell or Vitis Networking P4: implement RDMA-like direct network access on FPGA, bypassing host CPU for packet processing. Measure latency vs. CPU-mediated networking.
24. **Vitis AI evaluation** — run Vitis AI compiled model vs. custom HLS kernel on same workload, document resource/latency/power comparison, justify custom implementation
25. **Thermal-aware router integration** — FPGA temperature above threshold → reduce FPGA workload allocation in the router, measured response latency

### Deliverables
- `/fpga_engine/` with HLS kernels, TCL scripts, testbenches
- Synthesis reports: resource utilization, timing, power — for every kernel
- PCIe latency decomposition report
- SLR crossing penalty analysis
- SymbiYosys verification reports
- Partial reconfiguration demo
- Design doc: "FPGA Backend Architecture"

### Definition of done
All kernels meet Fmax targets. II=1 for all pipeline stages. SymbiYosys proves protocol correctness. Thermal-aware routing responds correctly to injected temperature conditions. FPGA network stack achieves lower latency than CPU-mediated path.

---

## Phase 8: TPU Backend
**Estimated duration: 3–4 months**

### Cloud infrastructure needed
GCP — apply for TPU Research Cloud (TRC) early (before this phase). TPU v4 or v5e VM. If TRC not approved, use paid GCP TPU VM (`v4-8` slice is cheapest entry point).

### What to learn first
- JAX fundamentals: `jit`, `vmap`, `grad`, `pmap`, pytrees
- XLA HLO: operations, shapes, layouts, tiling
- MLIR MHLO/StableHLO dialect: the bridge between ML frameworks and XLA
- TPU systolic array: how matrix multiply flows through the MXU without caches
- ICI topology: 3D torus, bandwidth vs. NVLink/EFA

### Build order
1. **GCP + TPU setup** — TPU VM, JAX installation, validate with simple JAX matmul
2. **TPU hardware deep dive** — benchmark MXU utilization via profiler, measure HBM bandwidth, document ICI latency/bandwidth vs. EFA
3. **MLIR → StableHLO lowering** — add a lowering pass that converts your runtime dialect ops to StableHLO ops, validate with `stablehlo-opt`
4. **StableHLO → XLA execution** — use JAX to execute the lowered StableHLO, validate outputs match CPU/GPU reference
5. **TPU memory layout optimization** — tile padding for systolic array alignment, measure MXU utilization before/after
6. **Explicit HBM ↔ SRAM scheduling** — understand that TPU has no hardware cache: the compiler (XLA) schedules all data movement. Document how your IR lowers to explicit data movement ops.
7. **pjit for distributed TPU** — automatic sharding across TPU chips, measure scaling efficiency on a multi-chip slice
8. **Multi-TPU collectives via ICI** — all-reduce using TPU-native ICI (not going through host), measure bandwidth vs. EFA all-reduce on equivalent GPU setup
9. **MXU utilization optimization** — ensure matmul dimensions hit 128×128 alignment, measure utilization %, document the performance cliff at non-aligned sizes
10. **VLIW ISA analysis** — inspect XLA-generated HLO for a kernel, understand the instruction bundling, document how this differs from x86 OOO and NVIDIA SIMT
11. **TPU Profiler integration** — capture TPU profiles, identify MXU utilization bottlenecks, HBM saturation, ICI contention
12. **TPU vs GPU cost model** — for each major op type: $/FLOP on TPU v4 vs. A100 vs. H100, FLOPS/Watt comparison, document when to choose each
13. **SparseCore (TPU v5)** — if using v5, implement embedding table lookup using SparseCore, compare against dense embedding on GPU

### Deliverables
- `/tpu_engine/` with lowering passes and benchmarks
- MLIR → StableHLO → TPU end-to-end demo
- TPU vs GPU benchmark comparison for matmul, attention, MLP
- Design doc: "TPU Backend Architecture and ISA Analysis"

### Definition of done
MXU utilization > 80% for large matmuls. End-to-end compilation pipeline works. Multi-TPU scaling measured. Cost model is calibrated and documented.

---

## Phase 9: Inference Serving
**Estimated duration: 2–3 months**

### Cloud infrastructure needed
GPU instance with large VRAM for serving experiments. TPU VM for TPU serving path.

### What to learn first
- vLLM paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- Speculative decoding: "Fast Inference from Transformers via Speculative Decoding" (Chen et al.)
- FlashDecoding: "FlashDecoding++: Faster Large Language Model Inference on GPUs"
- GPTQ: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

### Build order
1. **Paged KV cache** — block table abstraction, logical → physical block mapping, block allocator with free list, eviction policy. Validate correctness: same output as non-paged attention.
2. **Continuous batching** — dynamic request arrival, batch formation across variable sequence lengths, measure throughput improvement vs. static batching
3. **SLA-aware scheduler** — latency budget per request, preemption when budget exceeded, priority queuing
4. **FlashDecoding** — parallelize KV cache access across sequence dimension for long contexts, measure latency improvement for 8k+ token sequences
5. **Speculative decoding** — small draft model (e.g., 1B) + large verifier (e.g., 7B), token tree verification, acceptance rate measurement, throughput vs. latency tradeoff analysis
6. **GPTQ INT4 quantization** — group quantization with per-group scales, calibration dataset, measure perplexity vs. FP16/FP8/INT8 baselines, measure throughput improvement
7. **KV cache quantization** — INT8 KV cache, measure memory reduction vs. accuracy impact
8. **Backend-agnostic serving** — inference serving layer works with CPU, GPU, FPGA, and TPU backends via the unified router
9. **Serving benchmarks** — throughput (requests/sec), latency (p50/p99/p999 time-to-first-token, time-per-output-token), comparison against vLLM and TensorRT-LLM

### Deliverables
- `/inference_serving/` directory
- Throughput and latency benchmarks vs. vLLM and TensorRT-LLM
- Speculative decoding acceptance rate analysis
- Quantization accuracy vs. throughput tradeoff charts
- Design doc: "Inference Serving Architecture"

### Definition of done
Continuous batching achieves > 2x throughput vs. static batching. Speculative decoding achieves > 2x throughput on compatible workloads. GPTQ INT4 perplexity within 0.5 points of FP16.

---

## Phase 10: Observability, Testing, AI Integration
**Estimated duration: 3–4 months**

### Build order
1. **eBPF probes** — kernel scheduler (sched_switch, sched_wakeup), memory subsystem (page faults, huge page promotions), network stack (sock_sendmsg, tcp_retransmit). Use libbpf or BCC. Visualize as timeline traces.
2. **OpenTelemetry integration** — distributed tracing across all nodes, spans for every major operation, trace correlation with eBPF events
3. **Unified observability dashboard** — CLI report combining: latency histograms per backend, GPU utilization, FPGA temperature, collective throughput, memory usage per rank
4. **TLC model checking** — run TLC on all TLA+ specs (Raft, all-reduce protocol), verify no violations under all interleavings
5. **SymbiYosys CI integration** — formal verification runs on every FPGA RTL change
6. **Chaos test suite** — automated: inject network partition → verify recovery, kill GPU node → verify elastic resharding, inject FPGA thermal event → verify router response. All documented with recovery time measurements.
7. **Nsight profile analyzer agent** — ingests Nsight Systems `.nsys-rep` or Nsight Compute `.ncu-rep`, parses key metrics (SM utilization, memory bandwidth, warp efficiency), outputs ranked list of optimization suggestions with specific guidance. Runs automatically in CI on benchmark PRs.
8. **Kernel variant generator agent** — given a kernel specification (op type, input shapes, dtype), generates N variants with different tiling/vectorization parameters using Claude API, compiles and benchmarks each, promotes the winner. Documents the search process.
9. **LLM autotuning agent** — monitors runtime execution traces (latency per op, device utilization, memory pressure), identifies suboptimal placement or quantization decisions, proposes and tests changes. Compares pre- and post-agent optimization baseline.

### Deliverables
- eBPF + OpenTelemetry traces for an end-to-end multi-node training run
- TLC verification reports for all specs
- Agent demo: show the Nsight analyzer catching a real kernel inefficiency
- Agent demo: show the kernel variant generator finding a better tiling
- Design doc: "Observability Architecture"

### Definition of done
Agents demonstrably find real optimizations, not toy ones. eBPF traces show kernel scheduler and memory events correlated with application-level latency events. All TLC checks pass.

---

## Phase 12: Machine Learning Library
**Estimated duration: 3–4 months**
**Position in sequence:** After Phase 6 (distributed training infrastructure exists), before Phase 9 (inference serving). Classical ML uses Phase 1 lock-free primitives + Phase 2 SIMD. Deep learning builds on Phase 3 GPU kernels.

### What to learn first
- Bias-variance decomposition: the mathematical reason GBT overfits differently than a neural network
- Second-order optimization: why GBT uses Newton steps (Friedman 2001 — read the paper)
- PAC learning and generalization bounds: why they matter even when too loose to be practical
- Information-theoretic view of decision trees: entropy, Gini impurity, and why they differ
- The kernel trick: how SVMs achieve nonlinearity without explicit feature maps
- Ensemble theory: why bagging reduces variance but not bias; why boosting reduces both

### Phase 12a: Classical ML
1. **Decision tree (CART)** — Gini/entropy splitting, pre-pruning (max_depth, min_samples_leaf), post-pruning. Vectorized split search using SIMD.
2. **Random Forest** — bagging with bootstrap samples, `max_features = sqrt(p)` (classification) / `p/3` (regression), out-of-bag error estimation, feature importance via permutation. Parallelized over trees using work-stealing thread pool.
3. **Gradient Boosted Trees** — Friedman's algorithm with second-order (Newton) optimization, histogram-based split finding (LightGBM-style), column subsampling, L1/L2 regularization, shrinkage. Benchmarked against LightGBM throughput on large datasets.
4. **SVM** — SMO algorithm (Platt 1998), RBF/polynomial/linear kernels, kernel caching. Binary + multiclass (one-vs-rest).
5. **k-NN** — KD-tree (exact) and ball tree (approximate), SIMD distance computation, benchmarked against sklearn.
6. **k-means++** — improved initialization (Arthur & Vassilvitskii 2007), Lloyd's iteration with SIMD centroid update, elbow method for k selection.
7. **PCA** — incremental SVD (Halko et al. 2011 randomized algorithm), explained variance ratio, whitening. Benchmarked against sklearn.
8. **Linear models** — SGD with L1/L2 regularization (elastic net), L-BFGS. Logistic regression as baseline classifier.

### Phase 12b: Decision Framework + Benchmarks
9. **OpenML CC-18 runner** — automated evaluation harness: load all 18 datasets, train each algorithm, record accuracy + training time + inference latency.
10. **Cross-method comparison** — for each dataset: which model wins and why. Written analysis per dataset explaining the result in terms of data characteristics (size, dimensionality, categorical features, noise level).
11. **Decision criteria document** — written guide: given data type/size/constraint, which model family to reach for first and why. Covers: tabular vs. unstructured, small vs. large data, interpretability constraints, latency constraints.
12. **Hyperparameter sensitivity analysis** — for each algorithm: sweep key hyperparameters, plot accuracy vs. parameter, document the mechanism behind each sensitivity.
13. **Ensemble composition** — stacking (diverse base models + meta-learner), blending. Empirical validation that diversity is necessary (show correlated models add no value). Written rules for when to ensemble vs. not.
14. **Failure mode catalog** — for each algorithm: a concrete example where it fails badly and why. GBT on noisy labels, k-NN in high dimensions, SVM at scale, RF on highly imbalanced classes.

### Phase 12c: Hyperparameter Optimization
15. **Bayesian optimization** — Gaussian Process surrogate, Expected Improvement acquisition function, upper confidence bound. Benchmarked against random search and grid search.
16. **TPE (Tree-structured Parzen Estimator)** — Optuna-style, models p(x|y<threshold) and p(x|y>=threshold) separately. More scalable than GP-based BO.
17. **Hyperband / ASHA** — successive halving, early stopping of unpromising configs, async version for parallel workers.
18. **Population-based training** — exploit/explore schedule, mutation of hyperparameters mid-training. Useful for neural network training.

### Deliverables
- `/ml/` directory with all classical algorithms
- OpenML CC-18 benchmark results with analysis
- Design doc: "When to Use What" — decision criteria per scenario
- Design doc: "Algorithm Deep Dives" — mathematical foundations + failure modes for each method
- Ensemble recipe book with empirical validation
- Hyperparameter sensitivity charts per algorithm

### Definition of done
GBT matches LightGBM accuracy on OpenML CC-18 and beats it on training throughput for datasets > 1M rows. Decision framework document can correctly predict the winning algorithm for a held-out dataset before running it. Bayesian optimization finds better hyperparameters than random search in fewer evaluations (validated on 3+ datasets).

### Note on what this demonstrates
This phase demonstrates ML systems depth and empirical judgment. It does NOT replace: reading the original papers (Breiman 2001 RF, Friedman 2001 GBT, Platt 1998 SMO), studying learning theory, or breadth from diverse real-world problem experience. Read the papers alongside the implementations.

---

## Phase 11: Polish + Portfolio
**Estimated duration: 1–2 months (ongoing throughout)**

### Build order
1. **Public benchmarks** — end-to-end comparison against ONNX Runtime, XLA, TensorRT. Reproducible benchmark scripts. Published results with full methodology.
2. **Architecture diagrams** — system architecture, data flow, memory hierarchy per device, distributed topology
3. **RFC / design doc library** — `/docs/` folder with: scheduler RFC, memory model RFC, transport protocol RFC, compiler IR RFC, FPGA backend RFC. Each written to Google/Meta internal doc standard.
4. **Technical blog posts** — one per major component. Publish on personal site or Medium. Link from GitHub.
5. **Live demo script** — 30-minute walkthrough: start the system, run a training job, show latency histograms, show an agent optimization in action, show FPGA thermal response. Rehearse until smooth.
6. **MLSys submission** — identify the most novel contribution (likely: the heterogeneous router + MLIR dialect + FPGA network stack combination), write the paper, submit
7. **TRC application** (if not already done) — describe the project, apply for free TPU quota
8. **LLVM upstream** — if any MLIR bugs or improvements found, submit patches
9. **Python bindings (pybind11)** — expose tensor handle, inference engine forward pass, and benchmark harness to Python; pip-installable package structure; demonstrate running a full MLP forward pass from Python with latency matching the direct C++ call path

---

## Cross-cutting concerns (apply every phase)
- **Sanitizers**: every build runs ASan/TSan/UBSan. Zero tolerance for races or memory errors.
- **Benchmarking**: every new component gets benchmarked before moving on. No "I'll benchmark later."
- **Hardware counters**: every CPU benchmark includes IPC, cache miss rates. Every GPU benchmark includes Nsight metrics.
- **Design docs**: every non-obvious decision gets a written justification before implementation.
- **Performance regression CI**: benchmark results stored per commit, alert on regressions > 5%.

## Cloud cost estimate (rough)
| Phase | Instance | Est. monthly cost |
|---|---|---|
| Phase 3 (GPU learning) | g4dn.xlarge | ~$200 |
| Phase 3 (serious GPU work) | p4d.24xlarge (spot) | ~$800–2000 |
| Phase 5–6 (distributed) | 2x p4d.24xlarge (spot) | ~$1600–4000 |
| Phase 7 (FPGA) | F1 spot (~$0.50/hr) | ~$100 |
| Phase 8 (TPU) | TRC (free) or v4-8 | $0 or ~$400 |

Use spot instances aggressively for development. Reserve on-demand only for final benchmarks. Estimated total project cloud cost: $15,000–30,000 over 3–4 years at focused pace. Significant cost reduction possible with disciplined spot usage and development-only-when-needed discipline.
