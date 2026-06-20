"""
mxu_bench.py — measure MXU utilization % for various matmul dimensions
TODO: run on GCP TPU VM with JAX profiler

Expected: MXU util > 80% when M, N, K are multiples of 128 (TPU v4 MXU size).
Expected: significant perf cliff when dimensions are not multiples of 128.
"""
# import jax
# import jax.numpy as jnp
# import time

# sizes = [64, 128, 256, 512, 1024, 2048, 4096]
# for n in sizes:
#     a = jax.random.normal(jax.random.PRNGKey(0), (n, n), dtype=jnp.bfloat16)
#     b = jax.random.normal(jax.random.PRNGKey(1), (n, n), dtype=jnp.bfloat16)
#     f = jax.jit(jnp.matmul)
#     f(a, b).block_until_ready()  # warmup
#     t0 = time.perf_counter()
#     for _ in range(100):
#         f(a, b).block_until_ready()
#     elapsed = (time.perf_counter() - t0) / 100
#     tflops = 2 * n**3 / elapsed / 1e12
#     print(f"N={n:5d}: {tflops:.2f} TFLOPS")

print("mxu_bench: STUB — run on GCP TPU VM")
