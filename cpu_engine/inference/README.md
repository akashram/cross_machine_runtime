# Step 9 — CPU MLP inference engine

## What was built

A fully-connected multi-layer perceptron that runs with **zero heap allocation on the forward path**.

`mlp.h` provides:
- `MlpConfig` — layer dimensions and per-layer activation (`kRelu`, `kSigmoid`, `kNone`).
- `MlpInferenceEngine` — owns pre-allocated weights, biases, and two ping-pong scratch buffers. Constructor allocates everything; `forward()` never calls `operator new`.

## Design decisions

**Ping-pong buffers.** Two scratch buffers `buf_a_` and `buf_b_`, each of size `max(all layer dims)` floats, are allocated at construction. The forward pass swaps source and destination pointers each layer — no per-layer allocation, no reallocation.

**Fused bias + activation.** The existing AVX-512 elementwise kernels carry `__restrict__` on their in/out pointers. Rather than introducing a third temporary buffer, bias and activation are fused into a single in-place loop on the destination buffer after `matvec_f32` writes it. The compiler vectorises this loop.

**`matvec_f32` for the linear transform.** Uses the best available tier (AVX-512 on Linux, AVX2 auto-vectorised on macOS). Weight matrix is stored row-major: `W[m, n] = weights[m × in_dim + n]`.

## No-heap guarantee

Verified in `mlp_test.cpp` by overriding the four `operator new`/`delete` forms globally and counting allocations during 1 000 consecutive `forward()` calls. Both tested networks (router-size and deep MLP) produced **0 allocations**.

## Measured results (macOS Intel, AVX2)

| Network | Dims | Latency | Throughput | GFLOPS |
|---|---|---|---|---|
| Tiny | 64→128→64→32 | 3.4 µs | 295 K infer/s | 10.9 |
| Small | 256→512→256→128 | 34 µs | 30 K infer/s | 17.6 |
| Large | 1024→2048→1024→512→128 | 887 µs | 1.1 K infer/s | 10.8 |

## Key findings

**Small network is the compute-efficiency sweet spot.** At 17.6 GFLOPS, the 256→512→256→128 network has weight matrices small enough to fit in L2 (~256 KB), keeping the matvec compute-bound. The tiny network's weight matrices are all in L1 (≤ 64 KB total), so it's also fast but has fewer total FLOPs.

**Large network drops to 10.8 GFLOPS.** The 1024×2048 first weight layer is 8 MB — well beyond L3 on most Intel cores. The matvec becomes bandwidth-bound; each call streams 8 MB of weights from DRAM at ~17.9 GB/s (see step 10).

**Roofline context.** All MLP variants have arithmetic intensity AI ≈ 0.49 FLOP/byte (dominated by the weight matrix read). The ridge point on this machine is 3.73 FLOP/byte, so all MLP forward passes are **bandwidth-bound**. To cross the ridge they'd need batching (batch size B raises AI to ≈ 0.49 × B for large B) or weight reuse across requests.

## Platform notes

On Linux with AVX-512, the matvec path uses 512-bit FMA — roughly 2× the SIMD width, pushing toward the ridge point. IPC and L3 miss rates (step 11) will show whether the bottleneck is truly DRAM bandwidth or instruction throughput.
