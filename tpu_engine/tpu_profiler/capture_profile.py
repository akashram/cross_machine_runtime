"""capture_profile.py — capture a real TPU Cloud Profiler trace across a
representative training-shaped workload and flag the three bottleneck
signatures PLAN.md step 11 asks for: MXU utilization, HBM saturation, ICI
contention.

PLAN.md Phase 8 step 11: "capture TPU profiles, identify MXU utilization
bottlenecks, HBM saturation, ICI contention." Where steps 2/6/8 measure
each of these in isolation with purpose-built micro-benchmarks, this step
captures one combined trace of a workload that exercises all three at
once (a sharded MLP forward pass, same shape as pjit_distributed's step 7
kernel) — profiler-based bottleneck attribution on a realistic op sequence,
not another synthetic micro-benchmark.

Unrun here — no TPU device, and `jax.profiler.trace()` output needs
TensorBoard's profile plugin to actually inspect (this script captures the
trace directory; reading it is a manual TensorBoard step, not automated
here since it's an interactive UI, not a number this script could print).
"""

import time

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec as P


def sharded_mlp_workload(mesh: Mesh, batch: int, d_model: int, d_ff: int, steps: int):
    def mlp(x, w_up, w_down):
        h = jnp.maximum(x @ w_up, 0.0)
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
        sharded_mlp(x, w_up, w_down).block_until_ready()  # warmup / compile
        for _ in range(steps):
            out = sharded_mlp(x, w_up, w_down)
        out.block_until_ready()


def main(trace_dir: str = "/tmp/tpu_profiler_trace", batch: int = 4096,
          d_model: int = 4096, d_ff: int = 16384, steps: int = 20) -> None:
    n = jax.device_count()
    model_axis = 4 if n % 4 == 0 else 1
    data_axis = n // model_axis
    devices = mesh_utils.create_device_mesh((data_axis, model_axis))
    mesh = Mesh(devices, axis_names=("data", "model"))

    print(f"chip count: {n}, mesh: data={data_axis} model={model_axis}")
    with jax.profiler.trace(trace_dir):
        t0 = time.perf_counter()
        sharded_mlp_workload(mesh, batch, d_model, d_ff, steps)
        elapsed = time.perf_counter() - t0

    print(f"captured {steps} steps in {elapsed:.3f}s -> {trace_dir}")
    print(
        "\nOpen with: tensorboard --logdir=" + trace_dir + " (requires "
        "tensorboard-plugin-profile). In the trace viewer, check:\n"
        "  - MXU utilization: 'op_profile' tab, dot_general ops' device-time %\n"
        "  - HBM saturation: 'memory_profile' tab, peak HBM bytes vs. capacity\n"
        "  - ICI contention: 'trace_viewer' tab, look for gaps between "
        "collective ops and the compute they're supposed to overlap with "
        "(a gap means the sharded down-projection's implicit all-reduce, "
        "from pjit_distributed's step 7 sharding scheme, isn't hidden "
        "behind compute — the same overlap question hbm_sram's step 6 "
        "model addresses for HBM, just for ICI instead)."
    )


if __name__ == "__main__":
    main()
