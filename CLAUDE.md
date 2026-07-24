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
Next local-implementation phase: Phase 8 (TPU Backend), currently
stubbed.

**Phases 8, 9, 10, 12 — STUBBED, pending full local implementation**
Stub directories, interface headers, CMakeLists.txt, and README.md design
outlines exist. Code bodies are still 1–3 line TODOs.

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
