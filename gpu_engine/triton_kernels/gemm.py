"""Triton GEMM — the block-level-programming-model counterpart to
gpu_engine/kernels/gemm.cuh's naive -> tiled -> WMMA progression.

gemm.cuh needed three separate kernels to get from correct to fast:
naive (no reuse), gemm_tiled<TILE> (explicit __shared__ tiles + manual
__syncthreads() staging), gemm_wmma (explicit nvcuda::wmma fragment
load/mma/store calls, FP16-only, hard M/N/K alignment requirements).

matmul_kernel below is ONE kernel that plays all three roles depending on
what the Triton compiler decides for the target GPU:
  - `tl.dot(a, b)` is the single call standing in for gemm_tiled's manual
    __shared__ As/Bs staging AND gemm_wmma's fragment load/mma/store
    sequence. The programmer writes neither; `tl.dot` lowers to Tensor
    Core MMA instructions automatically whenever the block shapes and
    dtypes are Tensor-Core-eligible (fp16/bf16 inputs, dims that are
    multiples of the hardware's MMA tile shape) and falls back to a
    software-pipelined FMA sequence otherwise. gemm.cuh's variant 2 (tiled
    FP32) and variant 3 (WMMA FP16) are separate kernels for exactly this
    reason: `float` A/B never routes through wmma:: on any current
    NVIDIA architecture, so the .cuh had to hand-write both paths and the
    caller picks. `tl.dot` erases that fork at the source level, but the
    underlying hardware fork (Tensor Core eligible or not) is still real —
    it just now happens inside the compiler instead of inside the caller.
  - The K-loop's `tl.load` + shared-memory movement is compiler-managed
    software pipelining (double/triple buffering the next tile's load
    against the current tile's `tl.dot`), which gemm_tiled's single-buffered
    load -> syncthreads -> compute -> syncthreads loop does not do at all —
    that specific optimization (pipeline the load) is one gemm.cuh's README
    leaves on the table and this Triton kernel gets for free from the
    compiler pass, not from anything written here.

What is NOT free: block-size selection. BLOCK_M/BLOCK_N/BLOCK_K/GROUP_M
below are still a search space exactly like TILE=16 vs TILE=32 was for
gemm_tiled — `triton.autotune` automates the search, it does not remove
the need for one. And the alignment cliff gemm.cuh's WMMA variant documents
(M/N/K must be multiples of 16) has a Triton-level analogue for the
Tensor-Core-eligible path: `tl.dot` handles non-multiple-of-16 shapes
correctly via masking, but does NOT do so at Tensor Core throughput — the
same physical alignment cliff gpu_engine/precision/'s Tensor Core alignment
analysis (Phase 3 step 17) measured by hand shows back through here, just
without a crash if you miss it.

Status: HARDWARE-GATED, UNRUN. Same as elementwise.py.
"""

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Autotuning configs — same TILE-size search gemm.cuh's README documents
# by hand (TILE=16 for small/medium, TILE=32 for large), but swept
# automatically per (M, N, K) instead of picked once per problem size.
# -----------------------------------------------------------------------

_configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
                  num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
                  num_stages=4, num_warps=4),
]


@triton.autotune(configs=_configs, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped, L2-cache-locality-friendly program-id -> tile mapping
    # (the "swizzled" launch order from the Triton matmul tutorial —
    # gemm.cuh's row-major blockIdx.x/y grid has no equivalent of this;
    # it relies on whatever raster order the GPU's block scheduler uses).
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # This one call replaces gemm_tiled's inner unrolled dot-product
        # loop AND gemm_wmma's load_matrix_sync/mma_sync/store_matrix_sync
        # sequence — Tensor Core routing decided by the compiler, not here.
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B. A: (M, K), B: (K, N). Mirrors gemm_naive/gemm_tiled/
    gemm_wmma's call signature (M, N, K) minus the launch-grid boilerplate
    those need computed by hand via gemm_naive_grid/gemm_tiled_grid/
    gemm_wmma_grid — grid computation here is inline in the lambda below,
    same information, no separate helper needed."""
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
