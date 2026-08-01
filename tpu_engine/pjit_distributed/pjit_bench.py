"""pjit_bench.py — measure scaling efficiency of jax.jit + explicit sharding
across a multi-chip TPU slice.

PLAN.md Phase 8 step 7: "automatic sharding across TPU chips, measure
scaling efficiency on a multi-chip slice." Uses a 2D mesh (data x model)
matching the tensor/data-parallel split `distributed_training/`'s
`column_parallel_linear` and `tensor_parallel_attn` steps already implement
for GPU — same sharding shape, ICI instead of NVLink/EFA as the
interconnect.

`jax.experimental.pjit` was merged into `jax.jit` (an ordinary `jax.jit`
call now accepts `in_shardings`/`out_shardings`) in newer JAX releases;
kept as an explicit `pjit` import here since PLAN.md names it and older JAX
still exposes it separately — check which applies on the TPU VM's JAX
version.

Unrun here — no multi-chip TPU slice, and no local JAX install (this
repo's convention for hardware/toolchain-gated steps is real code the
target hardware would run, not a locally-runnable substitute).
"""

import time

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec as P


def build_mesh(data_axis: int, model_axis: int) -> Mesh:
    n_devices = data_axis * model_axis
    if jax.device_count() != n_devices:
        raise RuntimeError(
            f"mesh needs {n_devices} devices ({data_axis}x{model_axis}), "
            f"got {jax.device_count()}"
        )
    devices = mesh_utils.create_device_mesh((data_axis, model_axis))
    return Mesh(devices, axis_names=("data", "model"))


def sharded_mlp_step(mesh: Mesh, batch: int, d_model: int, d_ff: int):
    """One column-parallel-then-row-parallel MLP layer, matching
    distributed_training/column_parallel_linear + row_parallel_linear's
    GPU sharding scheme: up-projection's output feature dim is split
    across 'model', down-projection's input (contracting) dim is split
    the same way so no resharding communication happens in between —
    only the down-projection's output needs an all-reduce over 'model'."""

    def mlp(x, w_up, w_down):
        h = jnp.maximum(x @ w_up, 0.0)  # relu
        return h @ w_down

    sharded_mlp = pjit(
        mlp,
        in_shardings=(P("data", None), P(None, "model"), P("model", None)),
        out_shardings=P("data", None),
    )

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (batch, d_model), dtype=jnp.bfloat16)
    w_up = jax.random.normal(k2, (d_model, d_ff), dtype=jnp.bfloat16)
    w_down = jax.random.normal(k3, (d_ff, d_model), dtype=jnp.bfloat16)

    with mesh:
        out = sharded_mlp(x, w_up, w_down)
        out.block_until_ready()  # warmup / compile
        return sharded_mlp, (x, w_up, w_down)


def bench(sharded_fn, args, iters: int = 50) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        out = sharded_fn(*args)
    out.block_until_ready()
    return (time.perf_counter() - t0) / iters


def main(batch: int = 4096, d_model: int = 4096, d_ff: int = 16384) -> None:
    n = jax.device_count()
    print(f"chip count: {n}")

    # Single-chip-equivalent baseline: model axis = 1 (weights replicated,
    # data axis = n) gives the throughput ceiling scaling efficiency is
    # measured against.
    baseline_mesh = build_mesh(data_axis=n, model_axis=1)
    baseline_fn, baseline_args = sharded_mlp_step(baseline_mesh, batch, d_model, d_ff)
    with baseline_mesh:
        baseline_latency = bench(baseline_fn, baseline_args)
    flops = 2 * 2 * batch * d_model * d_ff  # up-proj + down-proj
    baseline_tflops = flops / baseline_latency / 1e12
    print(f"data={n} model=1 (baseline): {baseline_latency * 1e3:.3f} ms/iter, "
          f"{baseline_tflops:.1f} TFLOPS")

    # Sweep model-parallel factors that evenly divide the chip count.
    for model_axis in [f for f in [2, 4, 8] if f <= n and n % f == 0]:
        data_axis = n // model_axis
        mesh = build_mesh(data_axis=data_axis, model_axis=model_axis)
        fn, args = sharded_mlp_step(mesh, batch, d_model, d_ff)
        with mesh:
            latency = bench(fn, args)
        tflops = flops / latency / 1e12
        scaling_eff = 100.0 * tflops / baseline_tflops
        print(f"data={data_axis} model={model_axis}: {latency * 1e3:.3f} ms/iter, "
              f"{tflops:.1f} TFLOPS, {scaling_eff:.1f}% of baseline throughput")


if __name__ == "__main__":
    main()
