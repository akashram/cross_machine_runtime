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

**Phase 3: GPU Backend — CODE COMPLETE for the original 24 steps (2026-07-19);
steps 25-26 (Triton, CUTLASS) scoped 2026-08-09, not yet implemented — see
"Where we are: Phases 17-19" below.**
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

**Phase 6: Distributed GPU Training — CODE COMPLETE (25/25 steps, 2026-07-21)**
All 25 steps are code-complete and locally run on this Mac (`ctest`, real
captured numbers in each step's own README): data loading, data
parallel, grad accum, grad clipping, autograd engine + toy MLP, ZeRO-1/2/3,
ZeRO-Infinity offload scheduling, column/row-parallel linear, tensor-
parallel attention, sequence parallelism, 1F1B pipeline scheduling, 3D
parallelism, MoE/expert parallelism, checkpoint sharding, compute/comm
overlap, SyncBatchNorm, full training loop, 2:4 structured sparsity
training, supervised fine-tuning (SFT), reward model training,
PPO-based RLHF, and DPO. Portable — no CUDA/Linux dependency for any of
them; multi-rank steps use simulated ranks (real TCP loopback threads,
`networking/ring_allreduce` and `networking/collectives`). Only step 2
(GPUDirect Storage) stays hardware-gated (real cuFile API, code-complete,
unrun — no portable subset, see
`distributed_training/gpudirect_storage/README.md`). Step 22 (SFT)
trains the real `/transformer/` model on a toy instruction-tuning task
(masked next-token loss, 5 simulated data-parallel ranks), perplexity
18.3 -> 1.07 (see `distributed_training/sft/README.md`); it also added
`flatten_grad`/`unflatten_into_grad`/`accumulate_grad` to
`transformer/transformer_model.h` so `ModelGrads` can be all-reduced.
Step 23 (reward model) wraps the same transformer body with a scalar
reward head, trains with the Bradley-Terry pairwise objective on
preference pairs (5 simulated ranks), held-out ranking accuracy
0.667 -> 0.933 (see `distributed_training/reward_model/README.md`, which
also documents a real overfitting finding: holding out entire unseen
(a,b) prompts collapsed held-out accuracy to 0%, worse than random — the
dataset now splits at the preference-PAIR level instead, matching how
reward-model held-out eval normally works). Step 24 (PPO-based RLHF)
combines an SFT-initialized policy, a frozen reference copy, step 23's
frozen reward model, and a critic (`CriticParams` — literally
`RewardModelParams` reused, since a value function on a prompt is the
same architecture as a reward model on a prompt+response), trained with
the clipped PPO surrogate + a KL penalty against the reference folded
into the reward signal, data-parallel across 5 simulated ranks; mean
reward rises 0.93 -> ~2.9 over 10 iterations while mean KL stays bounded
(peaks at 1.26) — the reward-vs-KL tradeoff PLAN.md asks this step to
monitor, with no reward-hacking signature (see
`distributed_training/ppo_rlhf/README.md`). Step 25 (DPO) reuses
`bradley_terry_loss` and `policy_dlogits` unchanged — Rafailov et al.'s
reparameterization makes DPO's loss literally Bradley-Terry applied to an
"implicit reward" `beta*(log pi_policy - log pi_ref)` instead of a
learned reward model's output, so no new gradient math was needed. Uses
the identical SFT-init recipe and identical 75-pair preference dataset as
PPO for a real comparison; DPO's own loss/margin trace is clean and
monotonic (loss 0.637 -> 0.492, margin 0.121 -> 0.480) while a downstream
reward-model eval trace (matching PPO's metric) turned out too noisy to
trust at this toy scale (unconstrained reward-model scale + limited
headroom after a well-converged SFT policy) — documented as a real
finding, not hidden, with the pass criterion switched to the direct
loss/margin signal (see `distributed_training/dpo/README.md`, which also
has a DPO-vs-PPO comparison table and "when to prefer each" discussion).
Phase 6 is now fully code-complete; hardware validation is deferred along
with the earlier phases (see execution strategy below).

**`/transformer/` — minimal decoder-only transformer + tokenizer (added 2026-07-21, not one of the original 12 phases)**
Built specifically so Phase 6 steps 22-25 have a real model instead of
being stubbed or reduced to a toy classifier: real causal multi-head
attention (reuses `distributed_training/tensor_parallel_attn`), real
LayerNorm (reuses `distributed_training/seq_parallel`), real hand-derived
backprop through the whole stack, gradient-checked, and validated by an
actual training run that greedy-generates its training corpus back
exactly (`ctest -R transformer_test`). Character-level tokenizer, no
batching — see `transformer/README.md` for the stated scope. See
PLAN.md's "Minimal Transformer" section (inserted into Phase 6) for the
full rationale.

**Phase 7: FPGA Backend — CODE COMPLETE (25/25 steps, 2026-07-23)**
Lives in `fpga_engine/`. No AWS F1 instance on Mac, so — same split as
Phase 3/4 — every step is code-complete and locally runnable wherever it
doesn't strictly need Vivado/Vitis HLS/an FPGA card, with the
hardware-only piece written as real (not stub) TCL/HLS/XRT code, unrun,
clearly marked TODO in each step's README. Done so far: F1 setup
validation, TCL synth/impl/bitstream pipeline, Vivado power report CI,
AXI4-Stream passthrough, dot product II study, UNROLL/PIPELINE/DATAFLOW
comparison, DSP48E2 vs LUT tradeoff, ap_fixed precision/resource/latency
study, BRAM vs URAM access patterns, multi-bank DDR4 integration,
host-side DMA via XRT, PCIe latency decomposition, double-buffered
compute/transfer overlap, a fully pipelined INT8 MLP inference kernel,
timing closure critical-path analysis + retiming, SLR partitioning +
crossing penalty, clock gating dynamic-power modeling, and XADC die
temperature/voltage rail monitoring (step 18: `xadc_sensors.cpp` uses
XRT's real `get_info<thermal|electrical>()` sensor API, unrun; portable
`parse_xadc_json.py` parses the JSON and flags out-of-tolerance rails,
self-test passing locally — see `fpga_engine/xadc/README.md`), and an ILA
debug session on the AXI4-Stream interface (step 19: `ila_probes.tcl`
inserts a real ILA debug core on `axi_passthrough`'s TVALID/TREADY/TDATA/
TLAST nets via Vivado Hardware Manager, unrun; portable
`axi_trace_checker.py` mechanically applies the two AXI4-Stream handshake
rules an ILA session is normally read by eye, self-test catches a
synthetic free-running-counter protocol bug — see
`fpga_engine/ila_debug/README.md`), and cocotb testbenches for the AXI4-
Stream and DMA controller RTL (step 20: `fpga_engine/cocotb/` —
hand-written Verilog models driven by real cocotb tests against Icarus
Verilog, actually run locally, 4/4 passing — no F1/Vitis HLS dependency
for this step. Caught a real bug: the DMA controller's first version
sampled `mem_rdata` one cycle too early relative to a registered memory's
actual timing, producing a one-word-lagged copy; fixed by adding a second
read-wait state, confirmed by re-running the test. See
`fpga_engine/cocotb/README.md`), and SymbiYosys formal verification specs
for that same RTL (step 21: `fpga_engine/symbiyosys/` — `axi_formal.v`/
`axi_nodead.sby` prove the AXI4-Stream register slice's handshake always
resolves (VALID-hold + data-stability while stalled, II=1 latency, no
stuck backpressure), `dma_formal.v`/`dma_nooverlap.sby` prove
`mem_rden`/`mem_wren` are never both asserted, both via full k-induction
(`mode prove`) against the unmodified step-20 RTL. Actually run and
passing: `yosys` has no Homebrew bottle on this Mac (Tier 3 platform), so
rather than keep paying for a multi-hour from-source build (the same
shape of wall step 20's `python@3.12` install hit — `cmake`/`tcl-tk` also
building from source with LTO, ~2hrs in and still short of `yosys`
itself), switched to YosysHQ's prebuilt OSS CAD Suite release for
working `yosys`/`sby` in minutes; z3 (already installed) is the solver.
Along the way, found the free suite's `yosys` doesn't parse full SVA
`assert property (@(clk) disable iff (...) ...)` syntax (that grammar
needs the commercial Verific plugin — confirmed via the suite's own
bundled `fifo.sv` example, which has a Verific-gated branch using exactly
that syntax), so `axi_formal.v`/`dma_formal.v` were rewritten into
yosys-native procedural `assert`/`$past`/`$stable` form, same properties.
That surfaced two real formal-harness bugs (not RTL bugs) on first run —
an unguarded `$past()` producing a spurious counterexample from a
fictitious initial state, and an unconstrained `rst_n` letting the
solver start from an ungrounded garbage register state — both fixed
(`$initstate` guard; `initial assume(!rst_n)`) and both proofs now PASS
by full k-induction (basecase + induction both `pass`). See
`fpga_engine/symbiyosys/README.md`), and a Dynamic Function eXchange
(DFX) hot-swap flow (step
22: `fpga_engine/partial_reconfig/` — `dfx_pblock.tcl` defines a
reconfigurable pblock and implements two interface-compatible kernels as
its two configurations, `axi_stream/axi_passthrough.cpp` (RM_A, reused)
and new `axi_increment.cpp` (RM_B), writing a full bitstream for
config 1 and a partial bitstream for config 2 plus a `pr_verify` safety
check; `pr_host_driver.cpp` is the real XRT `load_xclbin()` hot-swap +
timing measurement. Both hardware-gated and unrun. `reconfig_time_model.cpp`
predicts hot-swap latency from partial-bitstream size — portable, run
locally: predicts 6.25ms for a small single-kernel RM at a modeled 400
MB/s ICAP bandwidth, for `pr_host_driver.cpp`'s real measurement to be
checked against once run. See `fpga_engine/partial_reconfig/README.md`),
and an RDMA-like FPGA-direct network path (step 23:
`fpga_engine/fpga_net/` — `rdma_bypass.p4` is a real P4_16 pipeline for
an OpenNIC-shell-style P4-programmable NIC: parses Ethernet/IPv4/UDP plus
a lightweight RDMA-style header and, on a WRITE opcode, dispatches
straight to a DMA-engine action in the same pipeline pass, no path that
hands the packet to host software before the payload lands;
`onic_shell_integration.tcl` wires the compiled pipeline into the
shell's user-plugin box. Both hardware/P4-toolchain-gated and unrun.
`net_latency_model.cpp` is the portable half, run locally: predicts
FPGA-bypass at 0.75us vs. CPU-mediated (kernel socket) at 10.30us for a
small one-sided WRITE, a 13.73x modeled speedup dominated by the
CPU-mediated path's kernel-stack-traversal + interrupt-dispatch +
context-switch stages the bypass path has no equivalent of — a
falsifiable claim `rdma_bypass.p4`'s real measurement can be checked
against once run. See `fpga_engine/fpga_net/README.md`), and a Vitis AI
DPU-vs-custom-kernel evaluation (step 24: `fpga_engine/vitis_ai/` —
`mlp_model.py` defines the same 16->32(ReLU)->8 MLP `ml_kernel/
ml_kernel.cpp` implements by hand, in PyTorch (Vitis AI's quantizer only
accepts framework models); `vai_compile_flow.sh` is the real
`vai_q_pytorch` calibration/deploy + `vai_c_xir` compile sequence that
would produce a DPU-deployable `.xmodel`. Both toolchain/hardware-gated
and unrun — no Vitis AI Docker image, DPU overlay, or F1 instance
locally. `dpu_vs_custom_model.cpp` is the portable half, run locally:
predicts the custom kernel at 163.3ns/inference (49 cycles at the 300MHz
`timing_closure/critical_path_model.cpp` already showed this exact
kernel's tree-retimed reduction closes) vs. a representative small
("B512"-class) DPU at 3506.7ns, a 21.5x modeled speedup dominated
(99.8%) by the DPU's fixed per-inference dispatch+weight-DMA overhead,
not compute — the same "shared engine pays fixed overhead a point-design
skips" argument step 23's networking model makes, and the
resource-footprint case (DPU's ~500 DSP/~50k LUT footprint is
workload-size-independent vs. the custom kernel's <=48 DSP and zero
BRAM/URAM) is the justification PLAN.md step 24 asks for. Caught a real
bug while writing it: mixing an `int` cycle-count sum directly into a
`%.0f` printf specifier is undefined behavior in C varargs and silently
corrupted two printed values; fixed by computing an explicitly-typed
`int` and using `%d`, confirmed by rebuilding with `-Wall`. See
`fpga_engine/vitis_ai/README.md`), and the thermal-aware router itself
(step 25: `fpga_engine/thermal_router/` — `thermal_policy.cpp` implements
`ThermalRouter::allocation_fraction_for_temp()`, the pure temperature ->
allocation-fraction decision logic (1.0/0.5/0.0 at the warning/throttle/
shutdown thresholds); `thermal_router.cpp` implements the hardware-
touching `read_fpga_temp_c()` via the same XRT `get_info<thermal>()` API
`xadc/xadc_sensors.cpp` uses, hardware-gated and unrun, sharing
`thermal_policy.cpp` so the real and locally-tested paths can never
diverge in their decision. `thermal_router_sim.cpp` is the portable half,
run locally: drives the router against a synthetic FPGA thermal event (a
first-order RC step response sampled at a 100ms poll interval) and
measures real response latency with `std::chrono` — 10.8ms at the
throttle threshold, 31.6ms at shutdown, both within the 100ms
poll-interval bound as they must be, while the router's own
decision-compute cost (4.29ns/call) is ~7 orders of magnitude smaller
than the poll interval, confirming polling cadence — not router logic —
is the lever for a tighter response-latency budget. This is a real
measurement of real code actually run, not a hand-computed estimate. See
`fpga_engine/thermal_router/README.md`. Several steps followed a
"portable model + hardware-gated kernel" split (e.g. `clock_gating/
clock_gating_model.cpp` predicts dynamic power reduction vs. duty cycle
locally; `timing_closure/critical_path_model.cpp` and
`slr/slr_crossing_model.cpp` do the analogous thing for their steps) —
each step's own README documents which half is measured vs. TODO.

Phase 7 is now fully code-complete (25/25); hardware validation is
deferred along with the earlier phases (see execution strategy below).

**Phase 8: TPU Backend — CODE COMPLETE (13/13 steps, 2026-07-27)**
Lives in `tpu_engine/`. No GCP TPU VM on this Mac, so — same split as
Phase 3/4/7 — every step is real, non-stub code: GCP TPU VM provisioning +
JAX matmul/correctness validation (step 1), MXU/HBM/ICI hardware
benchmarks (step 2, `mxu_util_bench.py`/`hbm_bandwidth_bench.py`/
`ici_latency_bench.py`), a real MLIR `DialectConversion` pass lowering the
`runtime` dialect (`compiler/dialect`) to StableHLO (step 3,
`StableHLOLowerPass.cpp` — covers every op except `gather`/`scatter`, left
as a genuine TODO rather than a shallow wrong version, and
`transfer`/`kernel_call`, deliberately left illegal since they're
placement/kernel-spec artifacts that shouldn't reach the TPU path;
toolchain-gated one level deeper than the rest of Phase 4 since StableHLO
pins to a specific LLVM commit, not a release), StableHLO execution via
`jax.export` validated against a numpy reference (step 4), pjit-sharded
MLP scaling (step 7, mirrors `distributed_training/column_parallel_linear`
+ `row_parallel_linear`'s GPU sharding scheme so scaling efficiency is
directly comparable across backends once both have real numbers),
gradient all-reduce over ICI (step 8, notes that the closest real in-repo
number — `networking/ring_allreduce`'s ~0.1 GB/s loopback bandwidth — is
an explicitly-documented overhead floor, not a fair EFA comparison), a
dense 128-boundary MXU utilization-cliff sweep (step 9), an HLO/compiled-
text dump script plus a written VLIW-vs-x86-OOO-vs-NVIDIA-SIMT analysis
(step 10 — the load-bearing difference: a bad TPU VLIW bundle-packing
decision has no hardware fallback, unlike OOO's reorder buffer or SIMT's
multi-warp latency hiding), a combined MXU/HBM/ICI profiler capture script
(step 11), and a SparseCore-vs-dense-gather embedding comparison (step 13,
TPU v5-only since SparseCore doesn't exist on v4; the SparseCore sketch
raises `NotImplementedError` rather than faking a callable, since
`jax-tpu-embedding`'s API is less stable than core JAX). Three steps need
no TPU/JAX at all and were **actually compiled and run locally** (`clang++
-O2 -std=c++17`, no CMake, same convention as `fpga_engine`'s `*_model.cpp`
files, real captured output in each README): `layout_opt`'s MXU tile-
padding utilization-ceiling model (finding: vocab-size misalignment is
nearly free, 99.9%, but small-batch padding is catastrophic — 0.8%
ceiling at batch=1 decode — an argument for batching decode requests in
Phase 9 independent of any TPU-specific measurement), `hbm_sram`'s
analytical HBM↔VMEM double-buffering overlap model (confirms cubic tiles
become compute-bound as they grow while tall-skinny/wide tiles stay
transfer-bound regardless, explaining XLA's preference for roughly-cubic
tiling), and `cost_model`'s TPU v4/A100/H100 $/FLOP and FLOPS/Watt
comparison (finding: all three land within ~5% on raw $/PFLOP-hour at
peak, so the real device-choice driver for a given workload is which
device's utilization ceiling it can actually reach, not the peak-vs-peak
ranking). See `tpu_engine/README.md`'s per-step status table for which
half of each step is measured vs. TODO.

Phase 8 is now fully code-complete (13/13); hardware validation is
deferred along with the earlier phases (see execution strategy below).

**Phase 9: Inference Serving — CODE COMPLETE (9/9 steps, 2026-07-27)**
Lives in `inference_serving/`. Unlike Phases 3/4/7/8, most of this phase
turned out to have no hardware dependency at all, so most steps are not
just written but **actually compiled, run, and tested on this Mac**, real
captured output in each step's own README: `paged_kv` (step 1,
stack-based `BlockAllocator` + per-sequence `BlockTable`, 300-trial
`foundation/proptest` property test — deliberately diverges from the
original stub's per-call device-pointer signature, since block
bookkeeping is backend-independent and device K/V bytes are a separate,
GPU-specific concern), `continuous_batching` (step 2, real
`ContinuousBatcher` class driving a discrete-event simulation against
static batching — 1.85x throughput speedup, 100% vs 52.8% device-slot
utilization, reproducing the vLLM paper's core finding as an actual
measurement), `sla_scheduler` (step 3, textbook-optimal
Earliest-Deadline-First scheduling with preemption falling directly out
of the admission rule — 0% SLA violations vs FIFO's 36.5% on a
mixed-urgency trace), `speculative_decoding` (step 5, reuses
`transformer/`'s real architecture as both draft and verifier, both
actually trained on CPU; the correctness property that matters — spec-
decoded output is token-for-token identical to plain verifier-only greedy
decoding regardless of draft quality — is verified directly, not just
claimed; 4.78x fewer verifier calls with a trained draft, 1.05x with a
near-random one, both exactly correct), `gptq` (step 6, real
Hessian-guided greedy column quantization — Gauss-Jordan Hessian
inversion, error compensation onto not-yet-quantized columns — applied to
a trained transformer's actual `w_out` weight with real calibration
activations; caught and fixed a real segfault from a weight-layout
mismatch during development, output MSE roughly half round-to-nearest's
on correlated calibration data), `kv_quant` (step 7, per-tensor symmetric
INT8 quantization round-tripped through K/V right after projection,
applied to the real trained model — exactly 4.000x memory reduction at
7B-class dimensions, perplexity impact unmeasurably small on this
saturated corpus), and `serving_backend` (step 8, `ServingRouter` with a
real CPU backend, GPU/FPGA/TPU registered with honest
`available = false` + reason strings rather than omitted, so the
GPU>TPU>FPGA>CPU fallback priority order itself is tested, not just "falls
back to whatever's available"). Two steps stay genuinely hardware-gated:
`flash_decoding` (step 4, real split-K CUDA kernels parallelizing
decode-step attention across the KV dimension — the axis
`gpu_engine/flash_attn`'s query-tile grid runs out of parallelism on once
there's only one query row per (batch,head) — unrun, no CUDA toolchain
here) and half of `serving_bench` (step 9 — this repo's own CPU-path
TTFT/TPOT/throughput numbers ARE real and measured locally, real
`std::chrono` timing of real transformer forward passes, with TPOT ~8.6x
TTFT as a direct, measured consequence of `transformer/model_forward`
having no KV cache; the vLLM/TensorRT-LLM comparison half stays
hardware-gated). Also fixed `inference_serving/CMakeLists.txt`'s
over-broad gate, which had been skipping the *entire* directory (not just
the GPU-dependent steps) unless `CMAKE_SYSTEM_NAME == Linux`. See
`inference_serving/README.md`'s per-step status table.

Phase 9 is now fully code-complete (9/9); hardware validation (step 4,
half of step 9) is deferred along with the earlier phases (see execution
strategy below).

**Phase 10: Observability, Testing, AI Integration — CODE COMPLETE (9/9 steps, 2026-07-27)**
Lives in `observability/`. Like Phase 9, most steps turned out to have a
genuinely local-runnable half, actually run with real captured output:
`opentelemetry` (step 2, hand-rolled span lifecycle + OTLP-JSON export —
deliberately not the OTel C++ SDK, same rationale as `foundation/proptest`
— parses its own exported JSON back and verifies 32/16-hex-char id formats,
parent-child linkage, attributes/events/errors), `dashboard` (step 3, real
ingestion/histogram/report pipeline run against synthetic-but-labeled
input shaped like `serving_bench`'s real numbers), `symbiyosys_ci`
(step 5, actually re-runs `fpga_engine/symbiyosys`'s two real passing
proofs with SHA-256 RTL-change detection — `yosys`/`sby` are already
installed from Phase 7, so this one runs for real unlike most of this
phase's toolchain-gated steps; caught and fixed a real bash-3.2
portability bug along the way), `chaos` (step 6, 2 of PLAN.md's 3 named
scenarios run for real — Raft leader-kill measuring actual 183.7ms
recovery time, reusing `networking/raft`'s real cluster; FPGA thermal
event using `fpga_engine/thermal_router`'s real decision logic — GPU-node-
kill stays a documented gap needing real GPU hardware), and the three
Claude-API-driven agent steps' portable halves (step 7's `ncu --csv`
parser, step 9's rule-based issue-detection heuristic — run against a
synthetic trace built from `serving_bench`'s own real numbers, correctly
recovering the same ~8.6x placement gap that step's analysis found by
hand). `ebpf` (step 1, real BCC tracepoint/kprobe program) and `tlc`
(step 4, real TLC model-checking configs for the existing Raft/Collective
TLA+ specs, including a proper MC-wrapper-module split to bound Raft's
otherwise-infinite state space) stay hardware/toolchain-gated — Java for
TLC was a deferred install decision, tracked alongside Phase 8's JAX
decision in project memory. All three AI-agent steps (7, 8, 9) get real,
complete Anthropic SDK client code (`model="claude-opus-4-8"`,
`output_config.format` structured outputs) but **no actual API calls are
made from this repo** — an explicit decision this session (see project
memory) to keep the three agent steps' LLM-calling halves gated the same
way GPU/TPU hardware is, rather than spend this session's API budget on a
benchmark-agent side task. Also fixed `observability`'s `CMakeLists.txt`,
which had been correctly building everything already — no gate bug here,
unlike `inference_serving`'s Phase 9 fix. See `observability/README.md`'s
per-step status table.

Phase 10 is now fully code-complete (9/9); hardware/toolchain validation
(steps 1, 4, the AI agents' real API calls, step 8's GPU half, chaos's
remaining scenario) is deferred along with the earlier phases (see
execution strategy below).

**Phase 12: Machine Learning Library — CODE COMPLETE (18/18 steps, 2026-07-28)**
Lives in `ml/`. Unlike Phases 3/4/7/8 (hardware-gated) but like Phases
9/10, every step here has no GPU/FPGA/TPU dependency at all, so all
18 steps are **actually compiled, run, and tested on this Mac**, real
captured output in each step's own README.

**Phase 12a: Classical ML (8/8)** — `decision_tree` (CART, Gini/entropy,
per-node feature subsampling for step 2's reuse), `random_forest`
(bagging, OOB error, permutation importance, `foundation`'s work-
stealing pool), `gradient_boosting` (Friedman 2001 Newton-step GBT),
`svm` (Platt 1998 simplified SMO, binary; `decision_function()` added
during Phase 12b for one-vs-rest wrapping), `knn` (KD-tree exact +
ball-tree exact/approximate, branch-and-bound verified against brute
force), `kmeans` (k-means++ measured ~24x lower inertia than random
init on 25 seeds), `pca` (randomized SVD, Halko et al. 2011; internal
linear algebra runs in `double` — see Phase 12b), `linear_models` (SGD
elastic-net + L-BFGS, `LinearModel::partial_fit`/`set_weights` added
during Phase 12c for PBT's warm-start). Every algorithm's own README
has real measured findings (bias-variance curves, sparsity effects,
failure-mode-adjacent behavior), not just "it runs."

**Phase 12b: Decision Framework + Benchmarks (6/6)** — `openml_bench`
fetches real datasets from the live OpenML REST API (`study/99` is the
actual CC-18 suite listing), selects and commits 8 of the 72 by size for
tractability, and runs all 6 supervised algorithms (KMeans/PCA excluded
as unsupervised) via a hand-written ARFF loader, no Python/sklearn
dependency; `cross_method` and `decision_criteria` are written analyses
grounded in those real numbers; `hyperparam_sensitivity` sweeps one
hyperparameter per algorithm against real data and **found and fixed a
real bug**: PCA's `explained_variance_ratio_` summed above 1.0 on
`wdbc`'s ill-conditioned raw features from float32 precision loss during
randomized-SVD power iterations — fixed by moving `pca.cpp`'s internal
linear algebra to `double` (public API unchanged, zero test
regressions); `ensemble` proves the diversity claim exactly on synthetic
data (disjoint-error models → 100% from 66.7%-accurate members) and
finds real-data nuance (naive majority voting can hurt a diverse
ensemble with unequal-strength members; a learned meta-model is more
robust); `failure_modes` demonstrates one concrete, measured failure per
algorithm (RF on imbalanced classes, GBT fitting label noise, SVM's
O(n²) scaling, KNN's curse of dimensionality, KMeans on non-convex
clusters, PCA discarding a low-variance discriminative direction,
DecisionTree instability, LinearModel on XOR).

**Phase 12c: Hyperparameter Optimization (4/4)** — `bayesian_opt` (exact
GP regression, Cholesky-based, `double`-precision internals; EI/UCB),
`tpe` (per-dimension Parzen-window density estimates, Bergstra et al.
2011 bandwidth heuristic), `hyperband` (Successive Halving + Li et al.
2016 bracket generation + Li et al. 2018 ASHA — **found and fixed a
real bug**: a rung with only 1 evaluated config would trivially
"promote" itself as top-1/eta of a group of one, fixed by requiring at
least `eta` results at a rung first, re-verified against a hand-
computable 4-config scenario with a known answer), `pbt` (Jaderberg et
al. 2017 truncation selection, backed by a genuine warm-start — added
`LinearModel::partial_fit`/`set_weights` rather than simulating
continuation). All four verify their core mechanism correct on
controlled synthetic cases (GP/EI/UCB monotonicity, TPE density/sampling
concentration, ASHA's exact promotion count, PBT's non-decreasing-best
invariant) and all four **independently find the same honest, real
result** on this repo's actual OpenML hyperparameter-tuning tasks: none
reliably beats a much simpler baseline (random search, or for PBT,
"random search without mid-training adaptation") — consistent with
Bergstra & Bengio 2012's published finding that random search is a
strong baseline on low-dimensional, wide-optimum landscapes, and
consistent with `cross_method`'s own finding that these particular real
datasets have accuracy ceilings algorithm/hyperparameter sophistication
doesn't move. Documented as a real, disclosed property of the datasets,
not papered over.

Phase 12 is now fully code-complete (18/18). **This completes every
phase in the original local implementation order (1→2→3→4→5→6→7→8→9→10→12)**
— see the execution strategy below for what's next (the hardware
validation pass, in the same phase order, one hardware type at a time).

**Phase 13: Retrieval-Augmented Generation — CODE COMPLETE (9/9 steps,
2026-07-29)**
Lives in `rag/`. Added 2026-07-28, not one of the original 12 phases —
same kind of addition as the Minimal Transformer note after Phase 6.
Unlike Phases 3/4/7/8 (hardware-gated), every step here has no GPU/FPGA/
TPU/Linux dependency at all, so **all 9 steps are actually compiled, run,
and tested on this Mac**, real captured output in each step's own README.
`embedding_model` (step 1, a bidirectional encoder — reuses
`seq_parallel`'s LayerNorm and `tensor_parallel_attn`'s attention
primitive unmasked, since `transformer_model.h`'s block hard-codes causal
masking — with mean pooling, a linear projection, and L2 normalize,
trained via symmetric InfoNCE/CLIP-style contrastive loss; gradient-
checked, in-batch retrieval accuracy 0.333 -> 1.000 after training; real
finding: `token_emb`'s finite-difference check needed a tighter epsilon
than every other parameter, traced to repeated characters within one
sequence causing a genuine second-order truncation effect, not a
gradient bug). `cosine_ann` (step 2, lives in `ml/knn/` alongside Phase
12a's `BallTree` — `CosineBallTree` L2-normalizes points once and
delegates to `BallTree` unchanged, since ranking by Euclidean distance on
unit vectors is exactly cosine-similarity ranking; verified directly
against brute-force cosine ranking, 0/20 queries mismatched). `hnsw`
(step 3, Malkov & Yashunin 2016, simplified neighbor-selection heuristic,
benchmarked directly against `BallTree` on the same corpus per PLAN.md's
own ask; real, honest finding: at n=2000/64 dims, `BallTree`'s EXACT
search visited fewer distance-evaluated points than HNSW's APPROXIMATE
search while also being exact — HNSW's asymptotic advantage shows up at
larger scale than this corpus, not assumed to always win). `indexing_pipeline`
(step 4, sentence-boundary-aware chunking with overlap + embed + index;
adds the shared `rag/corpus/corpus.h` fixture — 8 hand-labeled queries
plus, from step 6 onward, 40 deterministic distractor documents — reused
by steps 4/6/7/8; 100% top-1 retrieval accuracy end to end). `rag_generation`
(step 5, prompt construction + generation, composing step 4 with
`inference_serving::make_cpu_backend()` rather than a third
reimplementation of greedy decode). `recall_eval` (step 6, recall@1/3/5 =
1.000 exact vs. 0.875 recall@5 under `BallTree`'s approximate mode over a
48-document index — the distractor documents exist specifically because
8 documents made approximate and exact search literally identical, same
as `leaf_size` making a single-leaf tree in `ml/knn`). `generation_quality`
(step 7, a causal QA model trained to answer from real retrieved context
or abstain — generation accuracy 0.000 -> 1.000 with vs. without
retrieval, plus a ~14.7x teacher-forced-loss gap forcing the true answer
with vs. without context). `approx_retrieval_study` (step 8, pure
composition of steps 6/7's own `approximate` flags, zero new logic; real
finding: the recall drop from approximate retrieval is NOT free at the
system level — it costs exactly as much generation accuracy as the
recall drop itself, 0.125 both ways, and doesn't compound).
`serving_integration` (step 9, `route_rag()` wires retrieval + prompt
construction into `inference_serving`'s `ServingRouter` as a new request
path WITHOUT modifying `serving_router.h/.cpp` — Phase 9 stays usable
with zero Phase 13 dependency; GPU-preferred-with-fallback and
CPU-preferred-directly produce byte-identical output, confirming
`route_rag` genuinely delegates rather than duplicating dispatch logic).
See `rag/DESIGN.md` for the full design rationale, the recall/latency/
quality tradeoff measured end-to-end across steps 6-8, and two real,
disclosed toy-scale limitations (lexical- rather than semantic-similarity
contrastive learning in step 1; memorization- rather than
generalization-driven QA in step 7).

Phase 13 is now fully code-complete (9/9); no hardware validation is
needed for this phase (fully CPU-portable, unlike Phases 3/4/5/6/7/8/9's
GPU/FPGA/TPU/multi-node/eBPF pieces).

**Phase 14: Adversarial Robustness — CODE COMPLETE (8/8 steps including
the stretch goal, 2026-07-29)**
Lives in `adversarial/`. Added 2026-07-28, not one of the original 12
phases. Fully CPU-portable, no hardware gate anywhere: **all 8 steps are
actually compiled, run, and tested on this Mac**, real captured output in
each step's own README. Attacks `distributed_training::MLP` (the SAME toy
classifier `full_training_loop` already trains) via the generic
reverse-mode tape (`distributed_training/autograd.h`), not the
transformer — that tape already wraps its input batch in a `Tensor`
before the forward pass, exactly Goodfellow et al. 2014's original
continuous-feature setting, and attacking the transformer's discrete
token ids would need a structurally different (embedding-perturbation)
approach, a separate scope this phase didn't take on. `input_gradients`
(step 1, no engine modification needed — the generic tape never
special-cased parameter vs. input Tensors, so `x.grad()` already worked;
this step proves it directly via finite differences, median relative
error 0.0041, and fixes a real correctness detail: `input_gradient()`
zeros weight grads before every call so PGD's iterative calls can't
silently accumulate stale ones). `fgsm` (step 2, Goodfellow et al. 2014;
verified the L-infinity bound is exact and the attack genuinely
increases loss on a trained model). `pgd` (step 3, Madry et al. 2017;
verified PGD with `num_steps=1`/`step_size=epsilon` is byte-identical to
FGSM — a checkable structural relationship, not just argued — and 10-step
PGD finds a higher loss than one FGSM step at the same budget).
`vulnerability_measurement` (step 4, a clean accuracy-collapse curve on
an undefended model: 1.000 -> 0.956 -> 0.756 -> 0.544 -> 0.178 -> 0.011
FGSM as epsilon grows 0->4.0, PGD always at least as damaging).
`adversarial_training` (step 5, Madry et al.'s min-max formulation,
reusing `full_training_loop`'s forward->backward->clip->step shape
single-process; robust accuracy 0.722 -> 0.844 at `epsilon=1.5` over the
undefended baseline). `robustness_tradeoff` (step 6, pure composition of
steps 4/5, zero new attack/training logic; Tsipras et al. 2018's
robustness-costs-accuracy finding measured directly across a training-
epsilon sweep — no cost at `epsilon=0.5`, clean accuracy 1.000 -> 0.844 ->
0.811 as training epsilon grows to 2.5/3.5, while robust accuracy
improves at every epsilon without exception). `transferability` (step 7,
adversarial examples crafted against one model transfer to fool a
differently-sized/differently-initialized one at 37.8%, vs. only 2.2% for
a same-magnitude random-noise baseline — the 17x gap that establishes
genuine transfer, not just "large perturbations confuse any model").
`randomized_smoothing` (step 8, the stretch goal, REACHED not left as a
scope note — Cohen et al. 2019 certified defense with one disclosed
simplification, a normal/Wald confidence bound instead of the exact
Clopper-Pearson bound; every eval point certified with a substantial
average radius on this well-separated task, and a modest empirical gain
on PGD-perturbed inputs smaller than adversarial training's gain at a
comparable epsilon — consistent with smoothing's real value being the
certificate, not necessarily out-competing a targeted defense at its own
game). See `adversarial/DESIGN.md` for the full design rationale and two
disclosed toy-scale limitations (a 2-feature well-separated classifier
throughout; the Wald-vs-Clopper-Pearson approximation in step 8).

Phase 14 is now fully code-complete (8/8); no hardware validation is
needed for this phase (fully CPU-portable).

**Phase 15: NPU Backend — steps 2-7 CODE COMPLETE (6/7 steps not counting
step 1, 2026-07-30)**
Lives in `npu_engine/`. Same hardware-gated-but-locally-codeable pattern
as Phase 3/7/8: no AWS/GCP rental story for NPUs (edge/mobile hardware —
Apple ANE, Qualcomm Hexagon, Google Coral), so step 1 (CoreML/ONNX
Runtime NPU toolchain validation, a trivial op on real hardware) is
deferred to hardware validation — no ANE/Coral hardware and no
`coremltools`/`onnx`/`onnxruntime` installed locally (confirmed via
`ModuleNotFoundError` on both). Everything else is real, non-stub code,
actually built and run today: `quant_export` reuses
`inference_serving::GptqQuantizer` completely unchanged (at `bits=8`
instead of GPTQ's usual `bits=4`, since NPU toolchains default to INT8)
applied to `transformer/`'s real trained `w_out` weight, serializing to a
custom `.npuw` binary format — actually run locally, with the real
ONNX/CoreML writer (`npu_onnx_export.py`) left honestly unrun (same
Python packages missing); `cost_model` (`npu_cost_model.cpp`) is a
portable INT8 latency/efficiency model finding the NPU wins on power
efficiency (7.9 TOPS/W vs. GPU's 0.78 TOPS/W) and tiny-workload dispatch
overhead but loses to GPU on large-workload absolute latency — both
real, measured outcomes from the same model, not a one-sided "NPU wins"
narrative; `op_coverage` walks all 18 ops actually defined in
`compiler/dialect/RuntimeOps.td` and finds gather/scatter are the only
two architecturally excluded (dynamic indexing) — plus a separate real
finding that `PlacementPass.cpp` doesn't currently place gather/scatter
for ANY device, not just NPU, so the new NPU eligibility filter is
correct code that only becomes load-bearing once gather/scatter
placement exists at all; `thermal` mirrors `fpga_engine/thermal_router`'s
exact decision-logic/hardware-read split, disclosing a real platform gap
(no stable public per-ANE-block thermal sensor on Apple Silicon, unlike
FPGA's XADC); step 6 adds NPU as a device to `compiler/cost_model/
CostModel.cpp` (real edit, still builds+runs — the one Phase 4 file that
does) and extends `compiler/placement/PlacementPass.cpp`'s per-op device
eligibility (real edit, joins the existing MLIR-toolchain-gated unrun
bucket — unchanged status, not a new gap); step 7 registers NPU as a
fifth `ServingRouter` backend, `available=false`, placed last in the
fallback priority order (after FPGA, before CPU) since an
inference-only, edge/mobile-first, restricted-operator-model backend
isn't a peer to the three general datacenter accelerators already
registered. See `npu_engine/DESIGN.md` for the full rationale.

**Phase 16: Containerized & Orchestrated Deployment — 6/7 steps
code-complete (steps 1, 3-7; step 2 deferred, 2026-07-30)**
Cross-cutting, no single `*_engine/` directory — artifacts live at their
conventional real-world locations (`Dockerfile` at repo root, `k8s/` for
manifests), with `containers/README.md` and `containers/DESIGN.md`
holding the phase-level write-up. Every artifact is real and correct but
UNRUN: no Docker, kubectl, GPU, or Kubernetes cluster exist on this Mac.
Tracing the actual CMake dependency graph (not guessed) found the whole
tree only requires a C++23 compiler + CMake >=3.25 + Ninja — every
optional dependency (gRPC/FlatBuffers/libfabric/libbpf) is already
gracefully gated — and surfaced a plausible (unconfirmed) root cause for
`ci.yml`'s long-standing failures: it never pins `cmake`, and Ubuntu
22.04's apt version (3.22.x) predates this project's 3.25 requirement.
Step 1 (`Dockerfile`, multi-stage, portable subset) and step 3 (GPU
passthrough — `docker/gpu/daemon.json`, `Dockerfile.cuda`, compose file)
are real and unrun. Steps 4-5 (`k8s/serving/`, `k8s/training/`) are real
K8s manifests — a latency-driven HPA (not raw CPU%, per PLAN.md's
explicit ask) for serving, a `StatefulSet` with `podManagementPolicy:
Parallel` for gang-scheduled training — unrun (no kubectl/cluster).
Writing these manifests surfaced two real gaps, both now closed
(2026-07-30): no long-running `serving_daemon` process existed
(`inference_serving/serving_backend/` had only a unit test), and no
rank-per-process training driver existed (every multi-rank step in this
repo simulates ranks as threads in one process). `serving_daemon.cpp` is
a real long-running process wrapping `ServingRouter`, tested end-to-end
over real TCP (`nc`, real generated tokens back, clean SIGTERM shutdown)
— building it caught and fixed a real bug, an oversized prompt tripping
an `assert()` abort in `model_forward`'s positional-embedding indexing,
now clamped per-request instead of crashing the daemon.
`distributed_training/training_worker/` is a real process-per-rank
driver (`RANK`/`WORLD_SIZE`/`PEER_HOSTS` env vars, real inter-process
TCP via a new additive `TcpChannel` constructor overload in
`networking/common/channel.{h,cpp}` — the existing single-host
constructor and its test are untouched), validated as 4 actual separate
OS processes on this Mac: loss trajectory matches
`full_training_loop`'s thread-simulated baseline exactly (2.6317 ->
0.0008), real per-rank checkpoint shards written to disk. Both K8s
manifests now target these real binaries. Steps 6-7 (device-plugin gap
analysis: NVIDIA vs. Xilinx FPGA vs. TPU's node-pool model; custom
schedulers vs. Kubernetes, grounded in `topo_scheduler`/`multitenancy`'s
real measured behavior) are pure written analysis, no gate, done. Step 2
(cgroup interaction measured) is a bare one-line deferral rather than a
fake test plan — its entire deliverable is a real measurement that
needs Docker to produce honestly. Full `ctest`: 106/106 passing, zero
regressions from any of the above.

**Phases 17-19: scoped 2026-08-09, not yet started.** Added while comparing
this repo against a batch of ~10 job descriptions for analog/physics-based
AI compute roles (hardware architects, model architects, model-hardware
co-design, training infra, dynamical-systems theorists, SciML solver
engineers, analog circuit-sim engineers, PPA-modeling engineers). That
comparison found the repo deeply covers digital accelerator architecture,
distributed-training mechanics, compilers, and quantization/sparsity, but
has a clean zero on: analog/physics-based computing, non-volatile memory
devices, mixed-signal circuit modeling, PPA/dataflow accelerator-design
tools (Timeloop/Accelergy/CACTI-style), nonlinear dynamical systems (ODE/
SDE/PDE, stability, adjoint methods), the "unconventional" model
architectures those roles build instead of plain transformers (SSMs,
diffusion/flow, Neural ODEs, Deep Equilibrium Models, energy-based models,
muP scaling), and genuine PyTorch/JAX framework fluency (this repo's whole
training stack is hand-rolled C++ autograd). Three new phases close this,
full sections now in PLAN.md/SCOPE.md, following the same non-negotiable
convention as every phase above (real code, not stubs; run locally
wherever possible; honestly hardware-gated where not; README per step with
real measured numbers; design doc per phase):
- **Phase 17: Analog & Unconventional Compute Hardware** (`analog_engine/`,
  8 steps) — device non-ideality modeling, resistive-crossbar MAC
  simulation, NVM tradeoff comparison, analog-vs-digital energy modeling, a
  from-scratch PPA/dataflow tool (Weight-/Output-/Input-/Row-Stationary),
  a systolic design-space sweep on `transformer/`'s real GEMM shapes, a
  hardware-algorithm co-design case study reusing an existing repo
  algorithm, and an analog circuit transient surrogate. Unlike Phase
  7/8/15, there's no toolchain gate at all here — no analog/neuromorphic
  silicon exists to rent, so every step is fully local.
- **Phase 18: Dynamical Systems, SciML & Physics-Informed Architectures**
  (`sciml/`, 10 steps, the largest new phase) — ODE/SDE solver libraries
  (verified against closed-form/Monte-Carlo ground truth), Neural ODEs
  (adjoint-method gradients, finite-difference-verified like `adversarial/
  input_gradients`), Deep Equilibrium Models (implicit-function-theorem
  backprop), a state-space-model layer measured directly against
  `transformer/`'s attention, diffusion/flow-matching and energy-based
  generative models, a real muP hyperparameter-transfer measurement, and a
  noise-aware training bridge into Phase 17's device-noise model. Fully
  CPU-portable, no hardware gate.
- **Phase 19: Framework-Native Training (PyTorch/JAX)**
  (`framework_native/`, 6 steps) — the one gap that isn't about a new
  topic but about tooling: this repo demonstrates the underlying concepts
  (autograd, ZeRO-style sharding, data parallelism) entirely in hand-rolled
  C++, which doesn't show fluency with the actual industry-standard
  frameworks several JDs list as a minimum qualification. PyTorch and JAX
  ports of `transformer/`, `torch.compile` benchmarking, a real multi-
  process DDP run over CPU `gloo` diffed against `distributed_training/
  data_parallel`, a real FSDP run structurally compared against hand-
  written `zero1`/`zero2`/`zero3`, and one real run through a production
  framework (Lightning/DeepSpeed/Ray Train). PyTorch + Lightning +
  DeepSpeed + Ray approved for local install (2026-08-09), same precedent
  as the JAX install for Phase 8.

See `READING_LIST.md` for the full citation list backing all three phases
(analog/PPA-modeling papers, SciML/unconventional-architecture papers,
PyTorch/JAX framework references).

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
→ 8 → 9 → 10 → 12).** All of it is now code-complete (2026-07-28) — Phases
1, 2, 5 (partially), 6, 9, 10, 12 are also locally run with real captured
numbers; Phases 3, 4, 7, 8, and the remaining gated pieces of 5 are
code-complete but hardware-gated (see each phase's own status above).
Within each phase, implementation went step by step in PLAN.md's build
order. A step counts as implemented when it has real logic (not a stub)
and compiles wherever it can without the target hardware; benchmark
numbers stay TODO until the hardware validation pass.

**Phases 13-16, scoped 2026-07-28/29, are additions beyond the original
12-phase plan (see PLAN.md).** Same "write all local-implementable code
before hardware" philosophy applies to them: Phase 13 (RAG) and Phase 14
(Adversarial Robustness) are now BOTH code-complete and fully locally run
(2026-07-29, see above) — no hardware gate at all for either phase.
Phase 15 (NPU Backend) and Phase 16 (Containerized & Orchestrated
Deployment) are now BOTH code-complete for every locally-codeable piece
(2026-07-30, see above) — Phase 15 has only step 1 (real ANE/Coral
hardware + toolchain) left hardware-gated; Phase 16 has step 2 (cgroup
measurement, needs Docker), step 3 (GPU passthrough, needs GPU+Docker),
and steps 4-5 (needs a real Kubernetes cluster) left tool/cluster-gated.

**Next up: Phases 17-19, then the hardware validation pass below.** Every
phase in the original local-implementation order (1-10, 12-16) is
code-complete. Phases 17-19 (scoped 2026-08-09, see above) are the newest
addition to that same "write all local-codeable code before hardware"
queue — Phase 18 (SciML) and Phase 17 (analog/unconventional compute) are
both fully CPU-portable with zero hardware gate, so they're next; Phase 19
(framework-native training) follows once PyTorch/Lightning/DeepSpeed/Ray
are installed. After all three are code-complete, what's left across the
entire project is: (a) the hardware validation pass itself (GPU/FPGA/TPU/
multi-node/eBPF, phases 3-9 in original order, plus Phase 3's new
Triton/CUTLASS steps), blocked on provisioning cloud hardware; (b) Phase
15 step 1 (NPU/ANE hardware); (c) Phase 16 steps 2-5 (Docker/kubectl/
cluster, none installed locally — see the pre-hardware TODO in project
memory on the standing "no new local installs without asking" decision,
which now also covers Docker/kubectl alongside the existing JAX/Java-TLC
entries).

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
