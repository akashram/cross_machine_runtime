"""Benchmark driver for Triton elementwise + GEMM kernels vs. PyTorch's
native (cuBLAS/cuDNN-backed) ops, mirroring elementwise_bench.cu / gemm_bench.cu's
metric definitions (GB/s for elementwise, TFLOPS for GEMM) so the Results
tables in this step's README and gpu_engine/kernels/README.md are directly
comparable once both are filled in from real hardware.

Status: HARDWARE-GATED, UNRUN. Needs a CUDA GPU + `pip install torch triton`.
Run with: python bench_triton.py
"""

import torch
import triton

from elementwise import add, mul, relu, gelu, softmax
from gemm import matmul


def bench_elementwise():
    print("=== Elementwise: Triton vs torch (GB/s) ===")
    for n in (1_000_000, 16_000_000, 256_000_000):
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        ops = {
            "add": (lambda: add(a, b), lambda: a + b, 3),   # 2 reads + 1 write
            "mul": (lambda: mul(a, b), lambda: a * b, 3),
            "relu": (lambda: relu(a), lambda: torch.relu(a), 2),
            "gelu": (lambda: gelu(a), lambda: torch.nn.functional.gelu(a), 2),
        }
        for name, (triton_fn, torch_fn, bytes_per_elem_multiplier) in ops.items():
            bytes_moved = n * 4 * bytes_per_elem_multiplier
            t_ms = triton.testing.do_bench(triton_fn)
            r_ms = triton.testing.do_bench(torch_fn)
            t_gbs = bytes_moved / (t_ms / 1e3) / 1e9
            r_gbs = bytes_moved / (r_ms / 1e3) / 1e9
            print(f"  n={n:>11} {name:>5}: triton={t_gbs:8.1f} GB/s  torch={r_gbs:8.1f} GB/s")

    # softmax: 2D, separate table since it's row-wise
    for rows, cols in ((4096, 4096),):
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        bytes_moved = rows * cols * 4 * 2  # read + write
        t_ms = triton.testing.do_bench(lambda: softmax(x))
        r_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
        print(f"  softmax {rows}x{cols}: triton={bytes_moved/(t_ms/1e3)/1e9:8.1f} GB/s "
              f"torch={bytes_moved/(r_ms/1e3)/1e9:8.1f} GB/s")


def bench_gemm():
    print("=== GEMM: Triton vs torch/cuBLAS (TFLOPS, FP32) ===")
    for sz in (512, 1024, 2048, 4096):
        a = torch.randn(sz, sz, device="cuda", dtype=torch.float32)
        b = torch.randn(sz, sz, device="cuda", dtype=torch.float32)

        c_ref = a @ b
        c_triton = matmul(a, b)
        max_err = (c_ref - c_triton).abs().max().item()
        print(f"  M=N=K={sz}: max_abs_err={max_err:.2e} ({'PASS' if max_err < 1e-1 else 'FAIL'})")

        flops = 2 * sz ** 3
        t_ms = triton.testing.do_bench(lambda: matmul(a, b))
        r_ms = triton.testing.do_bench(lambda: a @ b)
        print(f"    triton={flops/(t_ms/1e3)/1e12:7.2f} TFLOPS  "
              f"torch(cuBLAS)={flops/(r_ms/1e3)/1e12:7.2f} TFLOPS")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "bench_triton.py requires a CUDA GPU"
    bench_elementwise()
    bench_gemm()
