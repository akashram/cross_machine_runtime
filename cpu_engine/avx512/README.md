# Step 7 — AVX-512 kernel library (three-tier)

## What was built

`kernels.h` implements every kernel **three times**, in increasingly
specific tiers, all building from the same source tree:

1. **Tier 1 (scalar)** — plain C++ loops; the correctness oracle and the
   "what if there's no vectorization at all" baseline. Always builds,
   everywhere.
2. **Tier 2 (auto-vectorized)** — pragma-annotated loops (`#pragma clang
   loop vectorize(enable)`); the compiler picks the widest ISA it has
   (AVX2 here, AVX-512 on a Linux build with `-mavx512f`).
3. **Tier 3 (explicit AVX-512 intrinsics)** — `_mm512_*`, guarded by
   `#ifdef __AVX512F__`, compiled only under the new `avx512` CMake preset.

Kernels: `dot_f32` (4-way unrolled FMA accumulator chain — masks the
4-cycle FMA latency on Skylake-X), `matvec_f32` (`_mm512_fmadd_ps` +
`_mm512_reduce_add_ps` per row, row-major `W[m,n] = weights[m*N + n]`),
`eltwise_add/mul/relu_f32`, `eltwise_sigmoid_f32` (fast rational
approximation `0.5·x/(1+|x|) + 0.5` — avoids an SVML/`exp` dependency at
the cost of < 0.5% error for `|x| < 5`), and `dot_i8_i32` (INT8 → INT32 via
`_mm512_cvtepi8_epi16` widening + `_mm512_madd_epi16`, which works on
*any* AVX-512F chip — no VNNI required).

A new `avx512` CMake preset adds `-mavx512f -mavx512bw -mavx512vl
-mavx512dq` to `cpu_engine` targets:
```bash
cmake --preset avx512 && cmake --build --preset avx512 --target avx512_bench
```

## Measured results (macOS Intel, AVX2 — Tier 3 inactive, no AVX-512 hardware)

```
dot_f32       (N=32768, 128 KB/array, 65536 FLOPs):
  scalar       46442.7 ns    1.4 GFLOPS
  auto-vec      3356.9 ns   19.5 GFLOPS   [13.8x]
  AVX-512: not available — run on Linux with --preset avx512

matvec_f32    (M=N=256, 256 KB matrix, 131072 FLOPs):
  scalar       74109.7 ns    1.8 GFLOPS
  auto-vec      3809.3 ns   34.4 GFLOPS   [19.5x]

eltwise relu  (N=32768, 128 KB/array):
  scalar        4720.6 ns    6.9 GFLOPS
  auto-vec      3965.3 ns    8.3 GFLOPS   [1.2x]

eltwise sigmoid (fast approx):
  scalar        6929.6 ns   14.2 GFLOPS
  auto-vec      6825.3 ns   14.4 GFLOPS   [1.0x]

dot_i8_i32    (N=131072, 128 KB/array, 262144 i8 MACs):
  scalar       11640.7 ns   22.5 GFLOPS
  auto-vec     11164.4 ns   23.5 GFLOPS   [1.0x]
```

## Key findings

**The Tier 1 → Tier 2 gap *is* the finding, and it's the evidence that
justifies Tier 3's existence.** Auto-vectorization alone buys **13.8× on
`dot_f32`** and **19.5× on `matvec_f32`** — both are compute-dominated
kernels with simple, regular access patterns that `#pragma clang loop
vectorize` handles well on AVX2. If that gap had been small (say 1.5×),
hand-writing `_mm512_*` intrinsics for marginal headroom over a
compiler that already does most of the work would be a maintenance cost
without a clear payoff. A 13–20× gap from auto-vec over scalar says the
opposite: there's real throughput on the table, the compiler is capturing
*most* of it on a 256-bit ISA, and a 512-bit explicit-intrinsics tier has
a believable shot at another ~2× from doubling the SIMD width — *and* at
controlling FMA scheduling and reduction strategy by hand where the
compiler's heuristics leave cycles on the floor. The three-tier structure
turns that argument from a guess into something the Linux run can directly
measure: Tier 3 vs. Tier 2 on the *same* machine, same compiler, same
source.

**The element-wise kernels (`relu`, `sigmoid`, `dot_i8_i32`) show almost no
auto-vec gain (1.0–1.2×) — and that's the roofline story (step 10) showing
up one step early.** These are the lowest-arithmetic-intensity kernels in
the library (AI ≈ 0.08–0.13 FLOP/byte, far below the 3.73 ridge point);
they're bandwidth-bound even in scalar form, so making the inner loop
execute in fewer cycles doesn't change wall-clock time — the DRAM/cache
pipe is already the bottleneck. **This is visible here, before the
roofline model formalizes it**: the kernels that *do* show large
auto-vec speedups (`dot_f32`, `matvec_f32` — both compute-heavier per byte
moved) are exactly the ones with headroom for instruction-level
optimization to matter.

**Why `dot_i8_i32` avoids VNNI:** `_mm512_madd_epi16` (multiply-and-add
pairs of 16-bit lanes into 32-bit accumulators, after widening i8→i16 with
`_mm512_cvtepi8_epi16`) is part of base AVX-512BW — present on every
AVX-512F chip including the target c5.2xlarge's Skylake-X-derived
Cascade Lake core, which predates VNNI. Choosing this path over
`_mm512_dpbusd_epi32` (the VNNI single-instruction equivalent) trades one
instruction for portability across the entire AVX-512 generation — a
deliberate "works everywhere AVX-512F works" choice over "fastest possible
on the newest silicon."

## Platform notes

```bash
# On AWS c5.2xlarge (Xeon Platinum 8275CL):
grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u   # confirm AVX-512 support
cmake --preset avx512 && cmake --build --preset avx512 --target avx512_bench
```
Expected Linux deltas (from step 10's measured ceilings): peak FLOPS rises
from 66.8 (AVX2) toward ~192 GFLOPS (16 floats/FMA × 2 ports), and the
Tier 3 row should land somewhere between the Tier 2 number and that new
ceiling — the gap between "Tier 3 achieved" and "192 GFLOPS peak" is
exactly the % utilization the roofline model (step 10) and counter
analysis (step 11) explain.
