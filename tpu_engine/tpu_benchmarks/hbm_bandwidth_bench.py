"""hbm_bandwidth_bench.py — measure achieved HBM bandwidth via a memory-
bound elementwise op, as a fraction of published peak.

PLAN.md Phase 8 step 2: "measure HBM bandwidth". A large elementwise add
reads two N-element buffers and writes one — 3*N*dtype_size bytes moved
per call — with negligible compute per byte, so the achieved GB/s is
HBM-bandwidth-bound rather than MXU-bound, unlike the matmul benchmark
in this same step.

Unrun here — no TPU device on this Mac.
"""

import time

import jax
import jax.numpy as jnp

# Published peak HBM bandwidth, GB/s. TPU v4: 32GiB HBM2 @ ~1200 GB/s.
TPU_V4_PEAK_HBM_GBPS = 1200.0

SIZES_ELEMENTS = [2**20, 2**22, 2**24, 2**26, 2**28]  # 1M .. 256M elements


def bench_elementwise_add(n: int, iters: int = 20) -> float:
    key = jax.random.PRNGKey(0)
    ka, kb = jax.random.split(key)
    a = jax.random.normal(ka, (n,), dtype=jnp.float32)
    b = jax.random.normal(kb, (n,), dtype=jnp.float32)

    f = jax.jit(lambda x, y: x + y)
    f(a, b).block_until_ready()  # warmup / compile

    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(a, b)
    out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / iters

    bytes_moved = 3 * n * 4  # read a, read b, write out; float32 = 4 bytes
    gbps = bytes_moved / elapsed / 1e9
    return gbps


def main(peak_gbps: float = TPU_V4_PEAK_HBM_GBPS) -> None:
    print(f"{'N (elements)':>14} {'GB/s':>10} {'% of peak':>10}")
    for n in SIZES_ELEMENTS:
        gbps = bench_elementwise_add(n)
        pct = 100.0 * gbps / peak_gbps
        print(f"{n:14d} {gbps:10.1f} {pct:9.1f}%")
    print("\nExpected: approaches peak at large N once launch/dispatch "
          "overhead is amortized; falls off at small N where the op is "
          "dispatch-bound, not bandwidth-bound.")


if __name__ == "__main__":
    main()
