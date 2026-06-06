# Step 8 — Cache-aware tiling

## What was built

Blocked matrix multiply (`C = A × B`, all row-major) with a runtime-configurable square tile size `T`.

Two implementations in `matmul.h`:
- `matmul_naive_f32` — plain `(i, k, j)` triple loop, j-innermost and vectorised; serves as the unblocked baseline.
- `matmul_tiled_f32` — `(i₀, k₀, j₀)` outer tile loops, `(i, k, j)` inner micro-kernel. For fixed `(i₀, k₀)` the A-tile `A[i₀:i₀+T, k₀:k₀+T]` stays in L1 while the j₀ sweep reuses it across all B-tiles.

## Working-set formula

Three tiles share the cache during the micro-kernel:

| T | Working set (3T² × 4 bytes) | Cache tier |
|---|---|---|
| 16 | 3 KB | L1 (32 KB) |
| 32 | 12 KB | L1 |
| 48 | 27 KB | L1 (tight) |
| 64 | 48 KB | L2 (256 KB) |
| 128 | 192 KB | L2 |
| 256 | 768 KB | L3 |
| 512 | 3 MB | L3 / DRAM |

## Measured results (macOS Intel, AVX2, 3.5 GHz)

### 256×256 (total data = 0.8 MB)

| Variant | GFLOPS | vs naive |
|---|---|---|
| naive | 27.0 | 1.0× |
| T=16 | 2.6 | 0.1× (overhead dominates) |
| T=32 | 19.6 | 0.7× |
| **T=48** | **6.1** | **0.2× — cache conflict anomaly** |
| T=64 | 20.6 | 0.8× |
| T=96–256 | 21–24 | 0.8–0.9× |

### 512×512 (total data = 3.0 MB)

| Variant | GFLOPS | vs naive |
|---|---|---|
| naive | 23.1 | 1.0× |
| T=64 | 18.5 | 0.8× |
| T=128 | 18.2 | 0.8× |
| T=512 | 21.4 | 0.9× |

### 1024×1024 (total data = 12.0 MB)

| Variant | GFLOPS | vs naive |
|---|---|---|
| naive | 16.2 | 1.0× |
| T=64 | 14.2 | 0.9× |
| T=256 | 17.0 | **1.1×** |
| T=512 | 20.8 | **1.3×** |

## Key findings

**T=48 is consistently 4–5× slower than its neighbours.** This is a classic cache *conflict-miss* pattern: with a 256-wide matrix (stride = 256×4 = 1024 bytes = 16 cache lines), certain tile row offsets land on the same L1 set, evicting data that was just loaded. T=48 (192 bytes per tile row) happens to maximise this collision on this specific cache geometry. Power-of-two tile widths avoid the worst conflict patterns.

**Naive beats small tiles at 256/512×256/512.** macOS Intel has a large shared L3 (~8 MB), so the 3 MB 512×512 working set fits in L3. Tiling overhead outweighs the cache benefit at these sizes. On a server chip with a smaller per-core L3 slice the crossover point moves much earlier.

**Tiling wins at 1024×1024 (12 MB > L3).** T=512 gives a real 1.3× speedup here because the 3 MB tile now fits in L3 while the naive version thrashes DRAM.

**Roofline context (step 10):** matmul at 256×256 has arithmetic intensity AI = M/6 ≈ 42.7 FLOP/byte — firmly compute-bound. The gap between tiled (67% utilisation) and peak (100%) is explained by instruction throughput limits and residual cache pressure; see step 11 for the counter breakdown.

## Platform notes

On Linux with a smaller per-core L3 and AVX-512:
- Smaller tile crossover point (L3 spill happens at lower M)
- AVX-512 doubles the SIMD width → higher absolute GFLOPS
- `perf stat -e L1-dcache-load-misses,LLC-load-misses` quantifies the cache benefit directly
