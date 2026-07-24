# vitis_ai — Vitis AI DPU vs. custom HLS kernel

**Status: the Vitis AI quantize/compile flow is code complete and unrun
(Vitis AI toolchain + DPU overlay + F1 required); the resource/latency
comparison model is measured locally.**

## What this measures
Three things, deliberately kept separate since two need the Vitis AI
toolchain and a DPU-loaded F1 instance neither of which exist locally:

1. **Resource/latency/power comparison model** (`dpu_vs_custom_model.cpp`):
   a first-order latency and resource budget for deploying the identical
   16 -> 32 (ReLU) -> 8 MLP (768 INT8 multiplies) two ways: through a
   Vitis AI-compiled DPU graph, vs. `ml_kernel/ml_kernel.cpp`'s hand-written,
   fully-pipelined HLS kernel. Compiles and runs on this Mac today — no
   Vitis AI/DPU/F1 needed.
2. **Model definition + quantization** (`mlp_model.py`): the same network
   shape as `ml_kernel.cpp`, defined in PyTorch (Vitis AI's quantizer only
   accepts framework models, not raw C++ arrays) so the two paths compare
   the same workload, not a stand-in.
3. **Compile flow** (`vai_compile_flow.sh`): the real Vitis AI CLI
   sequence — `vai_q_pytorch` calibration + deploy quantization, then
   `vai_c_xir` compilation against the target DPU's `arch.json` — that
   would produce a deployable `.xmodel` for the F1 shell's DPU overlay.

## Model caveats
Every DPU-side constant in `dpu_vs_custom_model.cpp` (PE array width,
clock, dispatch/DMA overhead, LUT/DSP footprint) is a representative
order-of-magnitude figure for Xilinx's DPU IP family in its smallest
standard configuration — the same kind of literature-sourced
approximation `fpga_net/net_latency_model.cpp` and
`clock_gating/clock_gating_model.cpp` use for their non-locally-measurable
stages, not a datasheet number for whatever specific DPU fingerprint an
AWS F1 Vitis AI shell would actually load. The custom-kernel side isn't
invented fresh either: its cycle count and 300MHz clock come directly
from two other portable models already in this repo —
`ml_kernel/ml_kernel.cpp`'s own pipeline structure (iteration counts) and
`timing_closure/critical_path_model.cpp`'s analysis of this exact
kernel's two reduction widths (tree depth and the 300MHz target it closes
under that model's optimistic assumption, after the tree-retiming fix —
see `timing_closure/README.md`; the unmodified *flat* kernel does not
close 300MHz per that same model). What doesn't depend on getting the
constants exactly right: a DPU is architecturally an instruction-driven
engine sized for large CNN/transformer layers, so it pays a fixed
per-inference dispatch + weight-DMA overhead a bare point-design kernel
has no equivalent of — structurally the same argument
`fpga_net/net_latency_model.cpp` makes about kernel-mediated vs. bypass
networking, and a falsifiable claim once `vai_compile_flow.sh` and
`ml_kernel.cpp` both have real F1 measurements.

## Results
**Resource/latency comparison model** (measured locally,
`clang++ -O2 -std=c++17 -Wall dpu_vs_custom_model.cpp -o dpu_vs_custom_model && ./dpu_vs_custom_model`):

```
=== predicted single-inference latency: DPU vs. custom HLS kernel ===
workload: 16 -> 32 (ReLU) -> 8 MLP, 768 total INT8 multiplies

custom (ml_kernel_mlp, tree-retimed @ 300MHz): layer1(32 iters+4 fill) + layer2(8 iters+5 fill) = 49 cycles = 163.3ns
custom peak DSP48E2 use: 32 (if layers share instances) - 48 (if not)

DPU (B512-class @ 300MHz): dispatch=2.0us + weight DMA=1.0us + readback=0.5us + compute=6.67ns (768 MACs / 512 MACs/cycle, rounds to 2 cycle) = 3506.7ns
DPU fixed resource footprint (order-of-magnitude, config-dependent): ~50000 LUT, ~500 DSP -- provisioned for large CNN layers, independent of this workload's size

predicted speedup = 21.5x (custom kernel), dominated by DPU dispatch+DMA overhead (3500.0ns of 3506.7ns, 99.8%) -- compute time itself is near-identical since 768 multiplies fits in a single DPU pass; the gap is architectural (instruction-driven shared engine vs. bare point-design pipeline), not arithmetic throughput.

resource verdict: custom kernel needs <= 48 DSP48E2 and 0 BRAM/URAM (both layers' operands are ARRAY_PARTITION-complete, i.e. registers, not memories) vs. the DPU's ~500 DSP + ~50000 LUT footprint that exists regardless of workload size -- for a network this small, embedded inside a larger pipeline (e.g. thermal_router/), the custom kernel's near-zero incremental footprint is the justification PLAN.md step 24 asks for; the DPU's advantage is reprogrammability across model shapes without resynthesis, which this single fixed MLP doesn't need.
```

An earlier version of this file mixed `int` sums directly into `%.0f`
printf specifiers — undefined behavior in C varargs (no implicit
int-to-double promotion happens there), which silently corrupted both
that value and the next argument in the list (`163 cycles` / `0.0ns`
instead of the correct `49 cycles` / `163.3ns`). Fixed by computing the
cycle count into an explicitly-typed `int` and using `%d`; rebuilding
with `-Wall` catches this class of bug going forward (it flagged the
original mismatch).

**Hardware** — TODO: run `vai_compile_flow.sh` against a DPU-loaded F1
shell and `ml_kernel/ml_kernel.cpp` synthesized on the same instance, and
fill in:

| Path | Latency (measured) | DSP | LUT | BRAM/URAM | Power |
|---|---|---|---|---|---|
| DPU (`vai_compile_flow.sh` output, via `vart`) | TODO | TODO | TODO | TODO | TODO |
| Custom (`ml_kernel_mlp`, tree-retimed) | TODO | TODO | TODO | TODO | TODO |

## Files
- `dpu_vs_custom_model.cpp` — portable, no Vitis AI dependency; run it directly.
- `mlp_model.py` — PyTorch definition of the same MLP shape, for the
  quantizer/compiler to consume. Requires PyTorch, unrun.
- `vai_compile_flow.sh` — the real Vitis AI CLI flow (quantize + compile
  to `.xmodel`). Requires the Vitis AI toolchain + DPU `arch.json`, unrun.

## Hardware notes
- Required: Vitis AI Docker image (`vai_q_pytorch`, `vai_c_xir`, `vart`),
  a DPU overlay bitstream loaded on the F1 card, AWS F1 instance.
- Once run: deploy `compiled/mlp_dpu.xmodel` through a `vart`-based host
  runner and time N forward passes the same way `pcie_latency/` times
  `ml_kernel.cpp`, for an apples-to-apples measured comparison.
