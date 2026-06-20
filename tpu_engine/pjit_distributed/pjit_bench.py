"""
pjit_bench.py — measure scaling efficiency with JAX pjit across TPU chips
TODO: run on GCP TPU VM
"""
# import jax
# import jax.numpy as jnp
# from jax.sharding import Mesh, PartitionSpec as P
# from jax.experimental import mesh_utils

# devices = mesh_utils.create_device_mesh((2, 4))  # 8 chips: 2x4 mesh
# mesh = Mesh(devices, axis_names=('batch', 'model'))
#
# @jax.jit
# def matmul(x, w):
#     return x @ w
#
# sharded_matmul = jax.pjit(matmul, in_shardings=(P('batch', None), P(None, 'model')),
#                            out_shardings=P('batch', 'model'))
#
# TODO: measure throughput and scaling efficiency vs single-chip

print("pjit_bench: STUB — run on GCP TPU VM")
