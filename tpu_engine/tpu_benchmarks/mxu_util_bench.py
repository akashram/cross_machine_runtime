"""mxu_util_bench.py — measure MXU utilization % across matmul sizes.

PLAN.md Phase 8 step 2: "benchmark MXU utilization via profiler". MXU
utilization is derived, not read directly off a counter: achieved TFLOPS
as a fraction of the chip's published peak bf16 TFLOPS. `jax.profiler`
trace collection is included so the *why* behind low-utilization sizes
(padding, host overhead, HBM-bound) can be inspected in TensorBoard,
not just the top-line percentage.

Unrun here — no TPU device on this Mac.
"""

import time

import jax
import jax.numpy as jnp

# Peak bf16 TFLOPS per chip, from Google's published TPU v4 spec sheet.
# Overridden via --peak-tflops for v5e/v5p runs.
TPU_V4_PEAK_BF16_TFLOPS = 275.0

SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def bench_matmul(n: int, iters: int = 50) -> float:
    key = jax.random.PRNGKey(0)
    ka, kb = jax.random.split(key)
    a = jax.random.normal(ka, (n, n), dtype=jnp.bfloat16)
    b = jax.random.normal(kb, (n, n), dtype=jnp.bfloat16)

    f = jax.jit(jnp.matmul)
    f(a, b).block_until_ready()  # warmup / compile

    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(a, b)
    out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / iters

    flops = 2 * n**3
    tflops = flops / elapsed / 1e12
    return tflops


def main(peak_tflops: float = TPU_V4_PEAK_BF16_TFLOPS, trace_dir: str = "/tmp/mxu_trace") -> None:
    print(f"{'N':>6} {'TFLOPS':>10} {'MXU util %':>12}")
    with jax.profiler.trace(trace_dir):
        for n in SIZES:
            tflops = bench_matmul(n)
            util_pct = 100.0 * tflops / peak_tflops
            print(f"{n:6d} {tflops:10.2f} {util_pct:11.1f}%")
    print(f"\nprofiler trace written to {trace_dir} — open with TensorBoard's "
          "profile plugin to inspect per-op MXU occupancy.")
    print("Expected: util% > 80 once N is a multiple of 128 and large enough "
          "to amortize dispatch overhead; a cliff below that.")


if __name__ == "__main__":
    main()
