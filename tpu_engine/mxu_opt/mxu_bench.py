"""mxu_bench.py — fine-grained sweep across the 128-boundary to characterize
the MXU utilization cliff at non-aligned matmul sizes.

PLAN.md Phase 8 step 9: "ensure matmul dimensions hit 128x128 alignment,
measure utilization %, document the performance cliff at non-aligned
sizes." Where tpu_benchmarks/mxu_util_bench.py (step 2) sweeps coarse
power-of-two sizes to characterize the MXU broadly, this step sweeps
densely around a single 128-multiple boundary (120..160, step 4) to show
the cliff shape directly — layout_opt/layout_opt_model.cpp already
predicts this cliff analytically (padding overhead); this script is the
real-hardware measurement that prediction should be checked against.

Unrun here — no TPU device on this Mac.
"""

import time

import jax
import jax.numpy as jnp

TPU_V4_PEAK_BF16_TFLOPS = 275.0

# Dense sweep straddling the 128 boundary: 120 (below), 128 (exact tile),
# 132/136/... (just past a tile, forces padding to 256).
SIZES = list(range(120, 161, 4))

REPEAT = 32  # batch the small matmul so a single call amortizes dispatch overhead


def bench_matmul(n: int, iters: int = 100) -> float:
    key = jax.random.PRNGKey(0)
    ka, kb = jax.random.split(key)
    a = jax.random.normal(ka, (REPEAT, n, n), dtype=jnp.bfloat16)
    b = jax.random.normal(kb, (REPEAT, n, n), dtype=jnp.bfloat16)

    f = jax.jit(lambda x, y: jnp.einsum("bij,bjk->bik", x, y))
    f(a, b).block_until_ready()  # warmup / compile

    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(a, b)
    out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / iters

    flops = REPEAT * 2 * n**3
    return flops / elapsed / 1e12


def predicted_ceiling_pct(n: int, tile: int = 128) -> float:
    padded = ((n + tile - 1) // tile) * tile
    return 100.0 * (n**3) / (padded**3)


def main(peak_tflops: float = TPU_V4_PEAK_BF16_TFLOPS) -> None:
    print(f"{'N':>5} {'TFLOPS':>10} {'measured util %':>16} {'predicted ceiling %':>20}")
    for n in SIZES:
        tflops = bench_matmul(n)
        measured_pct = 100.0 * tflops / peak_tflops
        predicted_pct = predicted_ceiling_pct(n)
        marker = " <-- tile boundary" if n % 128 == 0 else ""
        print(f"{n:5d} {tflops:10.2f} {measured_pct:15.1f}% {predicted_pct:19.1f}%{marker}")

    print(
        "\nCross-check against layout_opt/layout_opt_model.cpp's analytical "
        "ceiling: measured util % should sit at or below predicted ceiling "
        "% at every N (the ceiling is an upper bound from padding waste "
        "alone; real measured util is additionally limited by dispatch "
        "overhead, pipeline fill/drain, etc.)."
    )


if __name__ == "__main__":
    main()
