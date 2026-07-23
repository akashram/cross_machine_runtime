# ml_kernel — fully pipelined INT8 MLP inference

**Status: hardware synthesis is code complete and unrun (F1 required);
quantization accuracy is measured locally.**

## What this measures
Two things, deliberately kept separate since only one needs an F1
instance:

1. **Synthesis** (`ml_kernel.cpp`): a tiny 2-layer MLP (16 -> 32 ReLU ->
   8), fully unrolled on both layers' inner loops and pipelined
   (`PIPELINE II=1` on the outer loops) so one full forward pass issues
   every cycle. INT8 in/weights, INT32 accumulate. It elides the
   inter-layer requantization multiply a correct static-quantization
   scheme needs — it saturates the raw INT32 accumulator straight to INT8
   instead of rescaling by a calibrated activation scale first. That's a
   real simplification, not an oversight, and it's why (2) exists.
2. **Accuracy** (`mlp_int8_ref.cpp`): a portable, no-HLS-dependency
   reference that implements the *correct* scheme (explicit per-layer
   requantization, calibrated hidden-activation scale from a held-out
   calibration set) and compares it against float32 on the same network
   shape. This compiles and runs on this Mac today — no F1 needed — so
   its accuracy number is real, not a TODO, even though the hardware
   kernel's resource/timing numbers still are.

An earlier version of this file saturated the *final* layer's INT32
accumulator to INT8 range before dequantizing for comparison, which
destroyed the value being measured rather than quantizing it (RMS
relative error came out ~99%, i.e. noise). Leaving that accumulator
unsaturated until after the `* out_scale` dequantization fixed it — see
the comment at that point in `mlp_int8_forward` for why.

## Results
**Quantization accuracy** (measured locally,
`clang++ -O2 -std=c++17 mlp_int8_ref.cpp -o mlp_int8_ref && ./mlp_int8_ref`,
seed 42, 200 calibration samples, 2000 eval samples, weights/inputs ~
U(-1,1)):

| Metric | Value |
|---|---|
| RMS error vs. float32 | 0.0426 |
| RMS reference magnitude | 2.910 |
| Relative RMS error | **1.46%** |

**Hardware** — TODO: synthesize on F1.

| Kernel | II | Latency (cycles) | Fmax (MHz) | DSP | LUT | FF |
|---|----|----|----|----|----|----|
| `ml_kernel_mlp` | TODO | TODO | TODO | TODO | TODO | TODO |

CPU/GPU baseline comparison (PLAN.md asks for this): TODO — needs the
same MLP shape run through `cpu_engine`/`gpu_engine`'s inference paths
for a same-network latency comparison, not just an isolated FPGA number.

## Hardware notes
- Required for `ml_kernel.cpp`: AWS F1, Vitis HLS 2022.x
- `mlp_int8_ref.cpp` needs nothing but a C++17 compiler and runs anywhere
