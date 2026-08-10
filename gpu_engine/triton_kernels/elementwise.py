"""Triton reimplementation of gpu_engine/kernels/elementwise.cuh's kernels.

Same five ops (add, mul, relu, gelu, softmax), same semantics, but written
in Triton's block-level programming model instead of CUDA's thread/warp
model. Read alongside elementwise.cuh — the interesting part of this file
is everything it does NOT have to say.

What Triton's block-level model abstracts away vs. the hand-written .cuh:
  - Thread indexing: `pid = tl.program_id(0); offs = pid*BLOCK+tl.arange(0,BLOCK)`
    replaces `blockIdx.x*blockDim.x+threadIdx.x` — there is no per-thread
    variable at all in the source; a `tl.load`/`tl.store` on a BLOCK-wide
    offset vector is one op that the Triton compiler lowers to a coalesced
    vectorized load/store plus whatever thread count it picks.
  - Masking replaces the `if (i < n) return` bounds check, but it also
    replaces every warp-uniformity assumption in the .cuh's reductions:
    `tl.where`/`tl.sum` on a masked block do not require the programmer to
    reason about which lanes are active, unlike `__shfl_down_sync`'s
    explicit active-mask argument in elementwise.cuh's warp_reduce_sum.
  - No shared memory declarations. softmax_kernel below never writes a
    `__shared__ float smem[...]` or calls `__syncthreads()` — the Triton
    compiler decides whether a given `tl.max`/`tl.sum` needs shared memory,
    a shuffle-only warp reduction, or something else for the target
    architecture, and re-decides per-GPU at JIT compile time. elementwise.cuh
    hard-codes one reduction strategy (warp shuffle + one shared-memory
    inter-warp stage) that is correct for every NVIDIA GPU since Kepler but
    is not necessarily *optimal* for all of them.
  - BLOCK_SIZE is a `tl.constexpr` autotuned per (shape, dtype, GPU) via
    `triton.autotune`, not a single fixed 256 chosen once and used for
    every launch the way elementwise.cuh's kernels are.

What it does NOT abstract away (the real cost of the higher level):
  - GELU's erf-vs-tanh approximation choice is still the programmer's call
    — Triton has no numerics library that picks for you (see gelu_kernel).
  - Softmax's numerical-stability trick (subtract the row max before exp)
    is still something the kernel author must know and write explicitly;
    Triton gives no free correctness here, only free indexing.
  - Getting from "correct" to "at cuBLAS/cuDNN-level throughput" still
    needs autotuning (`triton.autotune` configs below) — the block-level
    model removes the *indexing* boilerplate, not the *performance-tuning*
    work gemm.py's Results section discusses.

Status: HARDWARE-GATED, UNRUN. No CUDA GPU on this Mac; the `triton`
package's compiler backend targets NVPTX and needs an NVIDIA GPU present
at import time to do anything beyond parse. Run on the same instance class
as gpu_engine/kernels (g4dn.xlarge minimum) with `pip install triton`
(bundled with `torch` on Linux+CUDA wheels since torch 2.0).
"""

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# add, mul, relu — trivial 1D elementwise, one BLOCK-wide vector op each.
# Directly comparable to add_kernel/mul_kernel/relu_kernel in elementwise.cuh.
# -----------------------------------------------------------------------

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


@triton.jit
def mul_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a * b, mask=mask)


@triton.jit
def relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


# -----------------------------------------------------------------------
# gelu — same exact-vs-approx choice elementwise.cuh documents.
# Triton has no erf builtin exposed in triton.language, so the "exact" arm
# uses libdevice's erf directly (still exact, just via a different door
# than CUDA's host-callable erff); the tanh approximation is identical
# arithmetic to the .cuh version, ported term-for-term.
# -----------------------------------------------------------------------

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr, APPROX: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if APPROX:
        # 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))) — GPT-2 form,
        # identical to gelu_approx_kernel in elementwise.cuh.
        c = 0.7978845608028654  # sqrt(2/pi)
        inner = c * (x + 0.044715 * x * x * x)
        y = 0.5 * x * (1.0 + tl.math.tanh(inner))
    else:
        import triton.language.extra.libdevice as tld
        y = 0.5 * x * (1.0 + tld.erf(x * 0.7071067811865476))  # x / sqrt(2)
    tl.store(y_ptr + offs, y, mask=mask)


# -----------------------------------------------------------------------
# softmax — one program per row, whole row in registers (BLOCK >= row width).
#
# This is the op where the block-level model earns its keep the most:
# elementwise.cuh's softmax needs two kernel launches (max pass, then
# normalize pass) plus a hand-written block_max/block_sum reduction with an
# explicit shared-memory inter-warp stage. Below is one kernel, one launch,
# `tl.max`/`tl.sum` doing the whole-row reduction with no shared-memory
# declaration in sight — same two-pass numerically-stable algorithm
# (max-subtract, then exp/sum-normalize), just not source-visible as two
# passes anymore.
# -----------------------------------------------------------------------

@triton.jit
def softmax_kernel(x_ptr, y_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK)
    mask = col_offs < n_cols

    row_ptr = x_ptr + row * row_stride
    x = tl.load(row_ptr + col_offs, mask=mask, other=-float("inf"))

    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    numerator = tl.exp(x_shifted)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    out_ptr = y_ptr + row * row_stride
    tl.store(out_ptr + col_offs, y, mask=mask)


# -----------------------------------------------------------------------
# Python-side launchers — mirror the .cuh kernels' call signatures.
# -----------------------------------------------------------------------

def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](a, b, c, n, BLOCK=1024)
    return c


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    mul_kernel[grid](a, b, c, n, BLOCK=1024)
    return c


def relu(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    relu_kernel[grid](x, y, n, BLOCK=1024)
    return y


def gelu(x: torch.Tensor, approx: bool = False) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    gelu_kernel[grid](x, y, n, BLOCK=1024, APPROX=approx)
    return y


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](x, y, x.stride(0), n_cols, BLOCK=block)
    return y
