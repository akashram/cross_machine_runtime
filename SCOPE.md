# Cross-Machine Runtime — Full Project Scope

> This document lists full feature scope by area, not build order. See
> PLAN.md's "Execution order" section: all phases get fully implemented in
> PLAN.md's phase order before any cloud hardware is provisioned; hardware is
> then used for a validation/tuning pass, in the same order, once every
> phase is code-complete.

## Core Runtime
- Lock-free task scheduler with work-stealing (Cilk-style, Chase-Lev deque)
- C++20 coroutine-based async execution engine
- Actor-style distributed execution model
- AOT graph capture and compilation
- Deterministic replay (record non-deterministic inputs, replay exactly for debugging)
- Fault tolerance / elastic training (checkpoint/restart, node failure, elastic resharding)
- Multiprocess programming / RPC
- C++23: concepts, constexpr metaprogramming
- Raft consensus for distributed control plane (leader election, membership changes, configuration updates) — TLA+ verified
- Chandy-Lamport distributed snapshots for consistent global state capture without stopping execution

## Multithreading
- False sharing prevention: cache-line padding, alignment-aware data structure layout
- `std::memory_order` semantics: acquire/release/seq_cst/relaxed — explicit usage and documentation of why each is chosen
- Lock-free data structures: SPSC and MPMC ring buffers, lock-free queues, lock-free freelists
- ABA problem: causes, solutions (tagged pointers, hazard pointers, epoch-based reclamation)
- Hazard pointers + epoch-based reclamation for safe memory reclamation in lock-free data structures
- RCU (Read-Copy-Update) for read-heavy shared data structures
- Work-stealing scheduler with Chase-Lev deque
- Deadlock prevention: lock ordering, lock-free-first design, documented locking discipline
- Race condition prevention: TSan in CI, documented ownership model per data structure
- POSIX thread primitives vs. C++ threading primitives — documented tradeoff
- Busy-poll / spinning vs. OS wait primitives — measured latency comparison
- CPU affinity / thread pinning to eliminate scheduler jitter on hot-path threads
- NUMA-aware thread pool: threads bound to NUMA node where their data lives
- Condition variables and futexes for off-hot-path coordination
- x86 TSO memory model vs. C++ memory model — documented relationship and implications for lock-free code

## Hardware Architecture — CPU
Deep understanding of CPU microarchitecture demonstrated through code, benchmarks, and design docs:
- Pipeline stages, out-of-order execution window (ROB size), and how to write code that exposes ILP
- Branch predictor internals (bimodal, TAGE): writing predictor-friendly code, measuring misprediction cost via perf counters
- Cache hierarchy: L1/L2/L3 sizes, associativity, set conflict avoidance, cache line size (64B x86)
- TLB structure and hugepage impact on TLB miss rate — measured
- Store buffer, write combining buffers, non-temporal stores (MOVNT) for write-only paths
- Store-load forwarding: when the CPU forwards from store buffer without hitting cache
- Execution units: FP/integer/load-store ports, throughput vs. latency for each instruction class
- Hardware prefetcher behavior: when it helps vs. hurts, software prefetch as complement
- CPU frequency scaling effects on TSC, turbo boost, and measured latency variance
- NUMA topology tools: `lstopo`, `numactl`, `likwid` — used as part of deployment tooling
- Hardware performance counters via `perf_event_open()`: cache misses, branch mispredictions, IPC, memory bandwidth utilization

## Hardware Architecture — GPU
Deep understanding of GPU microarchitecture demonstrated through kernels, profiling, and design docs:
- SM (Streaming Multiprocessor) structure: CUDA cores, Tensor Cores, warp schedulers, dispatch units, shared memory, register file
- Warp execution model: 32 threads per warp, SIMT, lockstep execution
- Warp divergence: causes, measurement (Nsight), avoidance strategies — documented per kernel
- Warp occupancy: threads/SM, registers/thread, shared memory/block — occupancy calculator, measured impact on throughput
- Warp scheduling to hide memory latency: latency hiding via sufficient in-flight warps
- Warp-level primitives: `__shfl_sync`, `__ballot_sync`, `__reduce_sync`, warp reduce patterns
- Cooperative groups: thread block, warp, grid scope synchronization
- Shared memory bank conflicts: 32 banks, stride patterns that cause conflicts, padding to avoid
- Global memory coalescing: aligned 128-byte transactions, access pattern analysis per kernel
- Register pressure: spilling to local memory, impact on occupancy — measured
- L1/L2 GPU cache behavior: cache line size, hit rate analysis via Nsight
- Instruction throughput vs. latency: understanding dual-issue, instruction mix optimization
- Hopper (H100) specific: Tensor Memory Accelerator (TMA) for async bulk memory copies, WGMMA (Warpgroup MMA) instructions for Tensor Core operations
- PTX/SASS inspection: reading generated assembly, identifying inefficiencies, writing PTX directly where needed
- Roofline model per kernel: compute-bound vs. memory-bandwidth-bound, % of hardware ceiling achieved

## Hardware Architecture — FPGA
Deep understanding of FPGA fabric architecture demonstrated through implementation and design docs:
- LUT structure (6-input LUTs on Xilinx UltraScale+), LUT packing and logic optimization
- DSP48E2 block structure: pre-adder, multiplier, accumulator, cascade chains — explicit use in kernels
- BRAM (36Kbit, true dual-port) vs. URAM (288Kbit, single-port) — access pattern tradeoffs per use case
- CLB (Configurable Logic Block) structure and packing efficiency analysis
- Routing resources: congestion analysis, placement constraints to reduce routing pressure
- Clock regions and clock domain crossing (CDC): proper synchronization, metastability avoidance
- SLR (Super Logic Region) structure in multi-die devices: SLR crossing penalty measured and documented
- DDR4 memory controller topology on UltraScale+ (F1): bank structure, row/column addressing, bandwidth saturation measurement
- PCIe hard IP block: understanding the endpoint, BAR mapping, interrupt mechanisms
- XADC: temperature sensors, voltage rails — programmatic monitoring
- Bitstream structure: partial reconfiguration boundaries, configuration time measurement
- SymbiYosys formal RTL verification: bounded model checking and k-induction for protocol correctness (e.g., AXI interface never deadlocks)
- Timing closure methodology: critical path tracing, retiming, placement-aware optimization, Fmax targets per design
- ILA (Integrated Logic Analyzer): probe insertion, trigger conditions, waveform capture for hardware debug

## Hardware Architecture — Memory
Deep understanding of memory hardware across the hierarchy:
- DRAM internals: row/column addressing, row activation (tRAS), precharge (tRP), CAS latency (tCL) — how these create the latency floor
- DRAM bank conflicts: how interleaved access patterns cause serialization, how to measure
- Memory channel interleaving: multi-channel bandwidth, how allocation affects channel utilization
- Cache coherence protocol: MESI/MOESI state machine, coherence traffic measurement, false sharing as a coherence artifact
- Write combining buffers: how they batch stores, how to exploit with MOVNT, how to avoid accidentally defeating them
- Non-temporal stores and loads: when to bypass cache entirely on write-only paths
- GPU memory hierarchy: HBM2 bandwidth, L2 cache, shared memory, register file — sizes and latencies for each level
- PCIe memory-mapped I/O: BAR space, MMIO vs. DMA tradeoffs
- Memory bandwidth saturation: measuring achieved vs. theoretical bandwidth via `likwid-bench`, STREAM benchmark, Nsight

## Hardware Architecture — Networking
Deep understanding of network hardware:
- NIC architecture: TX/RX descriptor rings, DMA for packet data, doorbell mechanism, completion notification
- RSS (Receive Side Scaling): hardware packet distribution across CPU cores, hash configuration
- RDMA NIC (RNIC) architecture: QP state machine, CQ, WR/SGE structure, inline data threshold
- PFC (Priority Flow Control): lossless Ethernet for RDMA, pause frame mechanism
- ECN (Explicit Congestion Notification): DCQCN congestion control for RoCE/EFA
- EFA hardware architecture: SRD transport, scalable reliable datagram, EFA-specific optimizations
- Hardware timestamping: NIC-level packet timestamps for sub-microsecond timing accuracy
- PTP / IEEE 1588: Precision Time Protocol for synchronized nanosecond-accurate clocks across distributed nodes — implemented and measured against NTP baseline
- PCIe topology: how the PCIe tree affects DMA bandwidth between CPU, GPU, FPGA, and NIC

## Hardware Architecture — TPU
Deep understanding of TPU microarchitecture demonstrated through implementation and design docs:
- Systolic array architecture: how data flows through the MXU (Matrix Multiply Unit) without caches, why it wins on large matmuls and loses on irregular ops
- MXU structure: 128×128 systolic array on TPU v4, bfloat16 operands, FP32 accumulation
- VPU (Vector Processing Unit): element-wise ops, activation functions, reductions — separate from MXU
- On-chip SRAM (HBM buffer): scratchpad-style, explicitly managed (no cache), compiler controls data movement
- HBM on TPU v4/v5: bandwidth, capacity, access patterns
- VLIW-like ISA: compiler-driven instruction scheduling with explicitly packed operation bundles — contrast with x86 out-of-order and NVIDIA SIMT
- ICI (Inter-Chip Interconnect): Google's proprietary high-bandwidth interconnect for TPU Pods, 3D torus topology
- TPU Pod topology: how chips tile into a Pod slice, ICI bandwidth vs. PCIe/NVLink/EFA comparison
- TPU v4 vs v5e vs v5p: architectural differences, when to choose each
- SparseCore on TPU v5: dedicated embedding table lookup hardware
- XLA HLO (High Level Operations): the IR that sits between frameworks and TPU hardware — understand how it maps to systolic array operations
- Profiling: Google Cloud TPU profiler, identifying MXU utilization, HBM bandwidth saturation, ICI contention

## CPU Backend
- AVX-512 vectorized kernels, manual SIMD intrinsics
- Cache-line aligned buffers, no heap allocs on hot path
- CPU affinity / thread pinning, NUMA-aware thread pool
- Busy-poll / spinning scheduler primitives (explicit vs. OS wait comparison)
- TSC-based nanosecond timing (RDTSC/RDTSCP)
- Hugepages (explicit, not THP — documented why)
- Prefetching (`__builtin_prefetch`)
- Branchless algorithms on hot path
- Non-temporal stores for write-only paths
- Profile-guided optimization (PGO): instrument, collect real workload profiles, recompile
- OS-level tuning as documented deliverable: `isolcpus`, `nohz_full`, IRQ affinity, C-state disabling, CPU frequency governor

## GPU Backend
- CUDA streams with compute/transfer overlap
- CUDA Graphs (capture + replay, near-zero CPU launch overhead)
- PTX/SASS inspection and optimization
- Custom Flash Attention kernel (tiled SRAM, beats cuDNN)
- Tensor Core alignment (dimension multiples of 8/16/32 per precision)
- Mixed precision: BF16/FP8 + FP32 master weights + loss scaling
- Warp-level primitives: `__shfl_sync`, `__ballot_sync`, warp reduce, cooperative groups
- Hopper-specific: TMA (Tensor Memory Accelerator) for async memory copies, WGMMA for Tensor Core operations
- SM occupancy optimization: register/shared memory tuning per kernel
- Warp divergence elimination: documented per kernel with before/after Nsight measurements
- Shared memory bank conflict elimination
- Global memory coalescing: verified per kernel via Nsight memory charts
- Roofline model analysis for every major kernel
- GPUDirect P2P (direct GPU-to-GPU without host staging)
- CUDA MPS for multi-tenant GPU sharing
- NVSHMEM evaluation vs. NCCL (documented tradeoff)
- Nsight Systems + Nsight Compute profiling
- NVML power measurement

## FPGA Backend
- Vitis HLS with full pipeline analysis (II=1 targets, loop pipelining, unrolling)
- AXI4 / AXI4-Lite / AXI4-Stream protocol variants
- DMA orchestration, interrupt vs. polling tradeoff (measured)
- DDR4 DRAM optimization (AWS F1, UltraScale+ VU9P, 4 DDR4 banks ~77 GB/s) — bandwidth measurement, multi-bank access patterns, memory controller utilization
- SLR partitioning (multi-die design, SLR crossing penalty measured)
- Partial reconfiguration (hot-swap kernels without stopping pipeline)
- FPGA network stack: RDMA directly on FPGA via OpenNIC / Vitis Networking P4 (bypasses host CPU entirely)
- Fixed-point arithmetic (ap_fixed<>, precision/resource/latency tradeoffs)
- DSP vs. LUT tradeoff per kernel (resources, latency, power)
- Clock gating for dynamic power reduction
- XADC temperature + voltage monitoring
- Vivado power reports as CI deliverable (every bitstream)
- Thermal-aware router: FPGA die temperature → automatic load redistribution
- Timing closure: critical path analysis, Fmax targets, retiming, placement constraints
- ILA (Integrated Logic Analyzer) for on-chip hardware debug
- Vitis HLS co-simulation (validate RTL against software model pre-synthesis)
- Cocotb Python-based RTL testbenches
- TCL scripting for fully headless, CI-able Vivado build pipeline
- Vitis AI evaluation (documented comparison vs. custom HLS with measurements)
- PCIe latency decomposition (each component measured separately)
- Ping-pong double buffering between DRAM and FPGA logic
- SymbiYosys formal RTL verification for AXI interface and pipeline correctness

## TPU Backend
- JAX/XLA as the primary programming interface for TPU
- MLIR dialect lowering to XLA HLO: how our IR compiles down to the TPU's native operation set
- TPU-specific memory layout: tile padding, dimension alignment for systolic array efficiency
- Explicit HBM ↔ SRAM data movement: understanding that the compiler must schedule transfers, there is no hardware cache
- Multi-TPU collective communication via ICI: TPU-native all-reduce without going through host — measured against EFA/NVLink baselines
- pjit / jit for distributed TPU execution and automatic sharding
- MXU utilization optimization: ensuring matmul dimensions saturate the 128×128 systolic array
- When TPU wins vs. GPU: large regular matmuls (TPU), irregular/sparse ops (GPU) — documented cost model
- TPU Profiler: MXU utilization %, HBM bandwidth utilization, ICI contention analysis
- GCP access: TPU v4/v5 VM instances (paid) + TPU Research Cloud (TRC) program application for free quota
- Integration with unified tensor handle and topology-aware scheduler
- TPU added to cost model: $/FLOP and FLOPS/Watt vs. GPU and FPGA

## Compiler / IR
- MLIR dialect design + lowering pipeline (not custom toy IR)
- MLIR Affine dialect / polyhedral model for loop optimization (tiling, interchange, fusion)
- Graph optimization, operator fusion, shape inference
- Kernel specialization and selection
- Memory planning (liveness analysis, buffer aliasing, peak-memory minimization)
- Rematerialization as MLIR pass (activation checkpointing at compiler level)
- Auto-sharding / GSPMD-style tensor placement (cost-model-driven)
- AOT compilation of execution graphs to native code
- Cost model for device placement decisions

## Memory Subsystem
- Unified tensor handle: device-local, pinned host, shared, FPGA DMA buffers, remote/distributed
- Per-device arena/slab allocators, lock-free freelists, size classes, alignment-aware
- Stream-aware GPU allocators (async frees, reuse after event completion)
- Unified transfer engine: CPU↔GPU, GPU↔GPU, CPU↔FPGA, GPU↔FPGA
- NUMA-aware allocation and thread binding
- Zero-copy IPC
- io_uring for checkpoint I/O path
- Memory-mapped files for fast checkpoint write/restart
- GPUDirect Storage: direct NVMe → GPU memory path without CPU staging — used for fast checkpoint loading and dataset prefetching at scale
- Intel RAPL CPU power measurement
- Explicit hugepages (2MB/1GB, documented tradeoff)
- Non-temporal stores for write-only paths bypassing cache
- Write combining buffer exploitation on CPU

## Distributed Layer
- 3D parallelism: data parallel + tensor parallel + pipeline parallel (Megatron-LM style)
- Compute/communication overlap (double-buffer backward pass + all-reduce)
- NCCL-like collectives: all-reduce, broadcast, reduce-scatter, all-gather
- Collective algorithm variants: ring all-reduce vs. recursive halving-doubling vs. tree all-reduce — measured comparison per topology and message size, documented decision model for when to use each
- Gradient compression (PowerSGD or 1-bit Adam)
- Tensor sharding, scatter/gather I/O
- Topology-aware scheduling (NUMA, PCIe, NVLink, rack, FPGA attachment paths)
- Tail latency mitigation: hedged requests (Google "Tail at Scale")
- Backpressure + load shedding protocol
- Vector clocks / Lamport timestamps for causal ordering
- TLA+ formal verification for at least one distributed protocol
- Raft consensus for control plane — TLA+ verified
- Chandy-Lamport distributed snapshots for consistent global state
- Chaos engineering harness (`tc netem` for simulated cross-region latency + packet loss, node failure injection)
- Multi-tenancy: resource quotas, priority preemption, fair scheduling

## Networking / Transport
- AWS EFA (Elastic Fabric Adapter): RDMA semantics over AWS's custom fabric via `libfabric` / OFI — same programming model as InfiniBand libibverbs (queue pairs, completion queues, registered memory, one-sided reads/writes, scatter/gather). Concepts transfer directly to InfiniBand; only physical fabric differs.
- RDMA one-sided vs. two-sided operations — documented tradeoff and measured latency comparison
- EFA SRD (Scalable Reliable Datagram) transport — AWS-specific, understand when to use vs. standard RDMA
- PTP / IEEE 1588: nanosecond-accurate clock synchronization across distributed nodes — implemented, measured vs. NTP baseline
- NIC descriptor ring architecture: TX/RX rings, DMA, doorbell, completion — understood and documented
- RSS (Receive Side Scaling): hardware packet distribution configuration
- PFC + ECN: lossless fabric configuration for RDMA (DCQCN)
- Kernel bypass: DPDK or AF_XDP
- gRPC + protobuf for control plane RPC
- Zero-copy serialization split: protobuf (control plane), flatbuffers/capnproto (data plane hot path)
- Custom userspace networking stack
- Anton 3 network (Shim et al., HPCA '22) as a design reference: study how DESRES achieved sub-microsecond all-to-all latency via custom ASIC network co-design — understand the tradeoffs vs. general-purpose EFA and what specialization buys at the extreme end

## Distributed GPU Training
- ZeRO optimizer stages (ZeRO-1/2/3): shard optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across data parallel ranks — measured memory reduction and communication overhead at each stage
- 1F1B pipeline schedule (interleaved 1-forward-1-backward): reduces pipeline bubble vs. naive GPipe — documented bubble fraction analysis and comparison
- Gradient accumulation: micro-batch accumulation before optimizer step, interaction with pipeline parallelism and ZeRO sharding
- Autograd engine: minimal reverse-mode autograd with tape-based gradient tracking, or explicit documented decision to interface with JAX/PyTorch autograd with measured overhead
- Flash Attention backward pass: custom tiled backward kernel alongside forward — memory footprint and throughput measured against naive attention backward
- Sequence parallelism: shard sequence dimension across ranks for long-context transformer training, composable with tensor parallelism
- Distributed optimizer state: Adam/AdamW optimizer with sharded momentum and variance tensors across data parallel ranks, consistent with ZeRO-2/3
- Gradient clipping: distributed gradient norm computation and clipping across ranks
- Distributed batch normalization (SyncBatchNorm): all-reduce of batch statistics across data parallel ranks
- Learning rate scheduling: warmup + cosine annealing with distributed-consistent step counting
- End-to-end training loop: forward → backward → gradient sync → optimizer step → checkpoint, with latency breakdown per phase
- ZeRO-Infinity: offload parameters, gradients, and optimizer states to CPU RAM and NVMe when GPU memory is exhausted — measured memory savings and throughput cost at each offload level
- Mixture of Experts (MoE) / Expert parallelism: learned routing mechanism dispatching tokens to experts on different devices — distinct from 3D parallelism, required for frontier model architectures (GPT-4, Mixtral style)
- Checkpoint sharding: sharded checkpoint format across ranks, async checkpoint writing overlapped with training, fast restore — measured checkpoint write/restore time at scale
- NCCL tuning: `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_BUFFSIZE`, `NCCL_SOCKET_NTHREADS` — topology-specific tuning with measured collective throughput before/after
- Ampere/Hopper 2:4 structured sparsity: hardware-accelerated 2 non-zero per 4 elements, 2x sparse matmul throughput on A100/H100 — model pruning to 2:4 pattern, measured throughput vs. dense baseline
- Distributed data loading pipeline: DataLoader workers, prefetching, dataset sharding across ranks, WebDataset format for streaming — measured GPU utilization with and without pipeline, ensuring data loading is never the bottleneck
- Supervised fine-tuning (SFT): instruction-response dataset, loss masking on prompt tokens, per-rank data sharding — establishes the policy initialization for RLHF
- Reward model training: preference pairs (chosen/rejected), Bradley-Terry objective, ranking accuracy measured on held-out preference set
- PPO-based RLHF: policy (SFT init) + critic (reward model) + KL penalty against frozen reference model, clip ratio, reward vs. KL divergence tradeoff measured across training, reward hacking monitored
- DPO (Direct Preference Optimization): offline alternative to PPO — optimize policy directly on preference pairs without a reward model; convergence speed and final reward compared vs. PPO baseline on identical preference data; documented decision guide for when to prefer each

## Inference Serving Layer
- Continuous batching
- Paged KV cache management (vLLM-style)
- SLA-aware request scheduling
- Speculative decoding: small draft model proposes tokens, large verifier model confirms in parallel — measured throughput gain vs. latency overhead
- GPTQ / INT4 quantization: weight-only INT4 with per-group scales — dominant LLM inference quantization method, measured perplexity vs. throughput tradeoff against FP8 baseline
- FlashDecoding: parallelizes KV cache access across sequence dimension during autoregressive decoding — distinct from Flash Attention forward pass, measured latency improvement for long-context inference
- LLM autotuning agent (final phase: after pre-AI optimized baseline exists)

## Observability / Profiling
- eBPF: kernel scheduler, network stack, memory subsystem probes
- OpenTelemetry: standardized distributed tracing (control plane)
- Flame graphs, timeline tracing, hardware counters
- Nsight Systems + Nsight Compute (including warp efficiency, occupancy, memory charts)
- Statistical benchmarking: confidence intervals, outlier detection, noise-controlled methodology
- Performance regression CI (automated benchmark comparison on every PR)
- NVML + RAPL power measurement
- Vivado power reports in CI
- Roofline model per kernel
- `perf_event_open()` hardware counters on CPU: IPC, cache miss rates, branch misprediction rates, memory bandwidth

## Testing / Quality
- ASan, TSan, UBSan in CI
- libFuzzer for IR parser/compiler frontend
- Property-based testing for distributed protocol invariants
- Cocotb testbenches for RTL
- SymbiYosys formal RTL verification
- TLC model checker for TLA+ specs
- Chaos engineering harness (doubles as a test suite)
- Statistical benchmark methodology

## AI Integration (Development Process)
- Nsight profile analyzer agent: ingests Nsight export, outputs ranked optimization suggestions, runs in CI
- Kernel variant generator agent: generates + benchmarks CUDA tiling/vectorization variants automatically
- LLM autotuning agent (end phase): observes runtime stats, adjusts placement/quantization decisions

## Cost / Management Layer
- Cost model: FLOPS/$ and FLOPS/Watt per device type
- AWS instance cost analysis: F2 vs. GPU instance tradeoffs per workload
- GCP TPU cost analysis: TPU v4/v5 vs. GPU instance tradeoffs per workload — $/FLOP and FLOPS/Watt comparison across all four backends (CPU, GPU, FPGA, TPU)
- Thermal-aware scheduling factoring cloud instance cost
- Design docs that frame system decisions in business/cost terms

## Portfolio / Career Deliverables
- `/docs`: RFCs and ADRs for scheduler, memory model, transport protocol (written as internal Google/Meta design docs)
- Technical blog post per major component
- Live demo script (runnable in 30-minute technical interview)
- MLSys conference submission when substantial
- LLVM upstream contribution (opportunistic — if bug/improvement found during MLIR work)
- TPU Research Cloud (TRC) program application — apply early, free TPU quota for qualifying projects
- CMake build system (modern, target-based, vcpkg/FetchContent)
- Public benchmarks vs. ONNX Runtime / XLA / TensorRT
- Architecture diagrams, latency profiles, tuning decision writeups
- Python bindings (pybind11): pip-installable package exposing tensor handle, inference engine forward pass, and benchmark harness — latency must match direct C++ call path
