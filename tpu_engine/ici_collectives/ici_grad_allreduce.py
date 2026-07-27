"""ici_grad_allreduce.py — gradient all-reduce over ICI for data-parallel
training, applying tpu_benchmarks/ici_latency_bench.py's raw-bandwidth
methodology to realistic gradient-tensor payload sizes instead of a
synthetic byte sweep.

PLAN.md Phase 8 step 8: "all-reduce using TPU-native ICI (not going through
host), measure bandwidth vs. EFA all-reduce on equivalent GPU setup." Where
step 2 (tpu_benchmarks/ici_latency_bench.py) sweeps arbitrary payload sizes
to characterize the interconnect itself, this step exercises the actual
collective a training loop issues: an all-reduce over a model's flattened
gradient, shaped like distributed_training/'s `flatten_grad` (see
transformer/transformer_model.h) — same op this repo's Phase 6 GPU/host
training steps do via networking/ring_allreduce, just over ICI instead of
EFA/TCP.

Unrun here — no multi-chip TPU slice, and no local JAX install.
"""

import time

import jax
import jax.numpy as jnp
from jax import lax


# Representative flattened-gradient sizes, matching real model scales this
# repo already trains (transformer/'s toy model is tiny; these are stand-ins
# for a small-to-mid LLM's parameter count so the collective's payload is
# realistic even though the model producing it isn't in scope here).
GRAD_SIZES_PARAMS = {
    "125M-param-class": 125_000_000,
    "1.3B-param-class": 1_300_000_000,
    "7B-param-class": 7_000_000_000,
}


def bench_grad_allreduce(n_params: int, iters: int = 20) -> tuple[float, float]:
    n_devices = jax.device_count()
    allreduce = jax.pmap(lambda g: lax.psum(g, axis_name="chips") / n_devices,
                          axis_name="chips")

    grad = jnp.ones((n_devices, n_params), dtype=jnp.bfloat16)
    allreduce(grad).block_until_ready()  # warmup / compile

    t0 = time.perf_counter()
    for _ in range(iters):
        out = allreduce(grad)
    out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / iters

    payload_bytes = n_params * 2  # bf16
    bytes_moved = 2 * (n_devices - 1) / n_devices * payload_bytes
    gbps = bytes_moved / elapsed / 1e9
    return elapsed, gbps


def main() -> None:
    print(f"chip count: {jax.device_count()}")
    print(f"{'model class':>20} {'params':>14} {'latency (ms)':>14} {'GB/s':>10}")
    for label, n_params in GRAD_SIZES_PARAMS.items():
        latency, gbps = bench_grad_allreduce(n_params)
        print(f"{label:>20} {n_params:14d} {latency * 1e3:14.2f} {gbps:10.2f}")

    print(
        "\nCompare against networking/ring_allreduce's real (loopback-TCP, "
        "not real-network) 0.1 GB/s effective bandwidth number -- that "
        "number is explicitly documented in its own README as overhead-"
        "dominated and not representative of real interconnect throughput, "
        "so it's a floor, not a fair EFA comparison point. A fair "
        "ICI-vs-EFA number needs both this step and networking/'s EFA "
        "steps run on real hardware; see tpu_benchmarks/README.md for the "
        "same caveat applied to the raw (non-gradient-shaped) bandwidth "
        "number."
    )


if __name__ == "__main__":
    main()
