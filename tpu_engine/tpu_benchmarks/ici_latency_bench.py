"""ici_latency_bench.py — measure ICI (Inter-Chip Interconnect) latency and
achieved bandwidth via an all-reduce across the TPU slice's chips.

PLAN.md Phase 8 step 2: "document ICI latency/bandwidth vs. EFA". Uses
`jax.pmap` + `lax.psum` so the collective runs over ICI directly, chip to
chip, never touching the host NIC — the property this step's README
compares against `networking/`'s EFA-based collectives, where the host CPU
issues every send/recv.

Unrun here — no multi-chip TPU slice on this Mac.
"""

import time

import jax
import jax.numpy as jnp
from jax import lax


def bench_psum_allreduce(payload_bytes: int, iters: int = 50) -> tuple[float, float]:
    """Returns (latency_seconds, achieved_gbps) for an all-reduce of
    `payload_bytes` per chip."""
    n_devices = jax.device_count()
    n_floats = payload_bytes // 4  # float32

    allreduce = jax.pmap(lambda x: lax.psum(x, axis_name="chips"), axis_name="chips")

    x = jnp.ones((n_devices, n_floats), dtype=jnp.float32)
    allreduce(x).block_until_ready()  # warmup / compile

    t0 = time.perf_counter()
    for _ in range(iters):
        out = allreduce(x)
    out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / iters

    # Ring all-reduce moves ~2*(N-1)/N * payload bytes per chip.
    bytes_moved = 2 * (n_devices - 1) / n_devices * payload_bytes
    gbps = bytes_moved / elapsed / 1e9
    return elapsed, gbps


PAYLOAD_SIZES = [4 * 1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024, 256 * 1024 * 1024]


def main() -> None:
    print(f"chip count: {jax.device_count()}")
    print(f"{'payload':>12} {'latency (us)':>14} {'GB/s':>10}")
    for size in PAYLOAD_SIZES:
        latency, gbps = bench_psum_allreduce(size)
        print(f"{size:12d} {latency * 1e6:14.1f} {gbps:10.2f}")

    print(
        "\nCompare against networking/'s EFA-based collectives "
        "(ring_allreduce, halving_doubling): those measure host-mediated "
        "RDMA over EFA between GPU nodes; this measures chip-to-chip ICI "
        "with no host CPU on the data path. Both are real-hardware-gated "
        "in this repo (no EFA NIC and no multi-chip TPU slice locally), so "
        "the comparison in tpu_benchmarks/README.md is spec-vs-spec until "
        "both get run."
    )


if __name__ == "__main__":
    main()
