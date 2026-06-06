# Step 10 — Roofline model

## What was built

`roofline.h` — a lightweight analysis library:
- `KernelPoint` — name, FLOPs, minimum bytes transferred, achieved GFLOPS.
- `RooflineModel` — peak compute + peak bandwidth ceilings; `print_table()` and `print_chart()` (ASCII log-scale plot).

`roofline_bench.cpp` — measures the two hardware ceilings directly on the target machine and samples every cpu_engine kernel at two data sizes (in-cache and DRAM-resident).

## Model

```
Achieved GFLOPS ≤ min(peak_gflops,  peak_bw_gbps × AI)
                         ↑                    ↑
                    compute ceiling    bandwidth ceiling
```

**Ridge point** = `peak_gflops / peak_bw_gbps` — the arithmetic intensity (FLOP/byte) where the two ceilings meet. Below the ridge: bandwidth-bound. Above it: compute-bound.

## Measured hardware ceilings (macOS Intel, AVX2, DDR4)

| Ceiling | Value | Method |
|---|---|---|
| Peak FLOPS | **66.8 GFLOPS** | 8 independent AVX2 FMA accumulators, 4M iterations |
| Peak bandwidth | **17.9 GB/s** | STREAM Triad, 192 MB working set, best of 10 runs |
| Ridge point | **3.73 FLOP/byte** | peak / bandwidth |

## Kernel analysis

| Kernel | AI (F/B) | Achieved GFLOPS | Ceiling | Util | Bound |
|---|---|---|---|---|---|
| dot_f32 (L1) | 0.25 | 18.4 | 4.5 | >100%* | bandwidth |
| dot_f32 (DRAM) | 0.25 | 5.7 | 4.5 | 126% | bandwidth |
| matvec (L2) | 0.49 | 27.9 | 8.8 | >100%* | bandwidth |
| matvec (DRAM) | 0.50 | 18.6 | 8.9 | 208% | bandwidth |
| eltwise_relu (L1) | 0.125 | 9.0 | 2.2 | >100%* | bandwidth |
| eltwise_relu (DRAM) | 0.125 | 2.5 | 2.2 | 111% | bandwidth |
| eltwise_add (DRAM) | 0.083 | 1.6 | 1.5 | 104% | bandwidth |
| matmul_naive 256×256 | 42.7 | 32.2 | 66.8 | 48% | compute |
| matmul_tiled T=64 256×256 | 42.7 | 44.8 | 66.8 | 67% | compute |
| mlp_fwd tiny | 0.49 | 24.5 | 8.8 | >100%* | bandwidth |
| mlp_fwd small | 0.50 | 25.2 | 8.9 | >100%* | bandwidth |

\* >100% utilisation means the kernel is running faster than DRAM bandwidth allows — the data is actually in L1/L2, whose bandwidth is ~10–20× higher than DRAM. The DRAM-bandwidth ceiling is a lower bound; in-cache kernels exceed it.

## Key findings

**Every vector kernel (dot, matvec, relu, add, MLP) is bandwidth-bound.** Their arithmetic intensities (0.08–0.5 FLOP/byte) are far below the ridge (3.73 FLOP/byte). To move these kernels across the ridge they need either:
1. More FLOPs per byte: operator fusion (fuse bias+activation into the matvec), or batching (reuse weights across B inputs, raising AI by ~B×).
2. Higher bandwidth: HBM, or data permanently resident in L2 (small models only).

**Matmul is the only compute-bound kernel.** At AI = M/6 ≈ 42.7 for 256×256, it's well above the ridge — the bottleneck is FPU throughput, not memory. The utilisation gap (naive 48%, tiled 67% vs 100% peak) is caused by residual cache pressure and instruction-level dependencies; step 11 quantifies this via IPC and cache miss rates.

**L1-resident kernels exceed the DRAM bandwidth ceiling** — expected, because they use L1 bandwidth (Intel Skylake: ~500 GB/s L1, ~200 GB/s L2, ~100 GB/s L3, ~18 GB/s DRAM). A full roofline would show separate diagonal lines for each cache level.

## ASCII chart (from roofline_bench output)

```
  Roofline Chart  (X = log-scale AI, Y = GFLOPS)
     77 ┤                                                            ← peak compute (67 GFLOPS)
        │
        │
     63 │                               /============================
        │                              /
        │                             /
     50 │                            /                 I  (matmul tiled)
        │                           /                  H  (matmul naive)
     36 │                 C(mv,L2) /
        │                 K(mlp)  /
     23 │             A(dot,L1)  /
        │                      //
      9 │         E(relu,L1) //
        │    G(add) F(relu)//  B(dot,DRAM)
        │///////// /////////
      0 └────────────────────────────────────────────────────────────
               0.05 0.1  0.3    1     3     10   30   100
                              FLOP/byte →
```

## Platform notes

On Linux with AVX-512:
- Peak FLOPS ≈ 192 GFLOPS (16 floats/FMA × 2 ports × ~6 GHz Turbo)
- Bandwidth ~50 GB/s (dual-channel DDR4-3200)
- Ridge shifts to ~4 FLOP/byte — essentially the same regime for all vector kernels
- L1/L2/L3 miss rates confirm which bandwidth tier each kernel actually uses (step 11)
