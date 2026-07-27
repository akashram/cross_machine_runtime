"""validate_matmul.py — sanity-check a fresh TPU VM: device visibility,
compiled matmul correctness, and a rough throughput number.

PLAN.md Phase 8 step 1's validation gate: "validate with simple JAX matmul".
Meant to run on the TPU VM `provision_tpu_vm.sh` creates. Unrun here — no
TPU device on this Mac, so `jax.devices()` would just return CPU.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np


def check_devices() -> None:
    devices = jax.devices()
    print(f"jax backend: {jax.default_backend()}")
    print(f"device count: {len(devices)}")
    for d in devices:
        print(f"  {d}")
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"expected backend 'tpu', got '{jax.default_backend()}' — "
            "this script is meant to run on a TPU VM"
        )


def check_correctness(n: int = 512, seed: int = 0) -> None:
    key = jax.random.PRNGKey(seed)
    ka, kb = jax.random.split(key)
    a = jax.random.normal(ka, (n, n), dtype=jnp.float32)
    b = jax.random.normal(kb, (n, n), dtype=jnp.float32)

    got = jax.jit(jnp.matmul)(a, b)
    want = np.matmul(np.asarray(a), np.asarray(b))

    max_abs_err = float(np.max(np.abs(np.asarray(got) - want)))
    print(f"matmul {n}x{n}: max_abs_err vs numpy = {max_abs_err:.3e}")
    if max_abs_err > 1e-2:
        raise RuntimeError(f"matmul correctness check failed: err={max_abs_err:.3e}")


def check_throughput(n: int = 4096, iters: int = 50) -> float:
    key = jax.random.PRNGKey(1)
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

    tflops = 2 * n**3 / elapsed / 1e12
    print(f"matmul {n}x{n} bf16: {elapsed * 1e3:.3f} ms/iter, {tflops:.1f} TFLOPS")
    return tflops


if __name__ == "__main__":
    check_devices()
    check_correctness()
    check_throughput()
    print("validate_matmul: PASS")
