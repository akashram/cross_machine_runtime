# Triton Kernels (Phase 3, Step 25)

**Status: STUB — requires CUDA GPU. Run on g4dn.xlarge → p3.2xlarge, same
instance class as `gpu_engine/kernels` (step 8-9), for a direct comparison.**
Added 2026-08-09 per the Phase 3 rescoping in PLAN.md/CLAUDE.md.

## What this measures

A Triton-language reimplementation of step 8/9's elementwise kernels
(add, mul, relu, gelu, softmax) and GEMM, benchmarked against both the
hand-written CUDA in `gpu_engine/kernels/` and PyTorch's native
cuBLAS/cuDNN-backed ops — three points of comparison, not two.

This step is a *written comparison* deliverable per PLAN.md: what Triton's
block-level programming model abstracts away vs. the thread/warp-level
control the hand-written kernels in steps 4-10 use directly.

## Files
- `elementwise.py` — add/mul/relu/gelu/softmax as `@triton.jit` kernels
- `gemm.py` — autotuned block-tiled GEMM using `tl.dot`
- `bench_triton.py` — benchmark driver, GB/s and TFLOPS metrics matching
  `gpu_engine/kernels/README.md`'s table format for direct comparison
- No CMake target: Triton is Python-only (`pip install torch triton`),
  same non-CMake precedent as `framework_native/`'s PyTorch/JAX steps —
  this is not a CUDA/C++ compilation unit.

## What the block-level model abstracts away

Full rationale is in the module docstrings of `elementwise.py`/`gemm.py`;
summary:

| Concern | Hand-written CUDA (`kernels/*.cuh`) | Triton (`elementwise.py`/`gemm.py`) |
|---|---|---|
| Thread indexing | `blockIdx.x*blockDim.x+threadIdx.x`, explicit per-thread scalar | `tl.program_id`+`tl.arange` over a BLOCK-wide vector, no per-thread variable in source |
| Bounds checking | `if (i < n) return` | `mask=` argument threaded through every `tl.load`/`tl.store` |
| Reductions (softmax) | Hand-written `warp_reduce_sum`/`block_max` with explicit `__shfl_down_sync` + one `__shared__` inter-warp stage, two kernel launches (max pass, normalize pass) | `tl.max`/`tl.sum` on a masked block, one kernel, no shared-memory declaration, compiler picks the reduction strategy per target GPU |
| Tensor Core dispatch (GEMM) | Two separate kernels: `gemm_tiled` (FP32, shared-memory only) and `gemm_wmma` (FP16-only, explicit `wmma::load_matrix_sync`/`mma_sync`/`store_matrix_sync`) | One `tl.dot` call; compiler routes to Tensor Core MMA or software FMA based on block dtype/shape at compile time |
| Software pipelining (GEMM) | Not implemented — `gemm_tiled`'s load→syncthreads→compute→syncthreads loop is single-buffered | `num_stages` in `triton.autotune` configs enables compiler-managed double/triple buffering for free |
| Block/tile size selection | Fixed per call site (`TILE=16` or `TILE=32`, picked by the programmer, documented in `kernels/README.md`) | `triton.autotune` searches a config list per `(M, N, K)` — automates the same search, does not eliminate it |
| Tensor Core alignment cliff | Hard assert-equivalent: `gemm_wmma` requires M%64==0, N/K%16==0 or it is simply wrong | `tl.dot` masks handle non-aligned shapes correctly but silently fall off Tensor Core throughput — same physical cliff `gpu_engine/precision`'s step 17 measured, now unenforced at the API level instead of crashing |

**The honest reading:** Triton removes indexing/reduction/dispatch
*boilerplate*, not the underlying hardware facts (alignment cliffs, the
value of pipelining, the need to search a tile-size space). Every real
optimization gemm.cuh's README calls out by hand — cache tiling, Tensor
Core alignment, occupancy — still has to be true of the compiled Triton
kernel to get comparable throughput; the language just stops making the
programmer spell out the mechanism.

## Results

TODO: run on GPU hardware and fill in this table (same metric definitions
as `gpu_engine/kernels/README.md` for direct comparison).

### Elementwise throughput (GB/s)

| Kernel | N=1M (Triton / torch) | N=16M (Triton / torch) | N=256M (Triton / torch) |
|--------|------------------------|--------------------------|----------------------------|
| add | TODO / TODO | TODO / TODO | TODO / TODO |
| mul | TODO / TODO | TODO / TODO | TODO / TODO |
| relu | TODO / TODO | TODO / TODO | TODO / TODO |
| gelu | TODO / TODO | TODO / TODO | TODO / TODO |
| softmax (4096x4096) | TODO / TODO | — | — |

### GEMM throughput (TFLOPS, FP32)

| Size | Triton | torch (cuBLAS) | gemm.cuh tiled (from kernels/README.md) | gemm.cuh WMMA (from kernels/README.md) |
|------|--------|------------------|--------------------------------------------|---------------------------------------------|
| 512 | TODO | TODO | TODO | TODO |
| 1024 | TODO | TODO | TODO | TODO |
| 2048 | TODO | TODO | TODO | TODO |
| 4096 | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU (Triton's NVPTX backend); Volta+ for `tl.dot` to
  route through Tensor Cores at fp16/bf16
- Install: `pip install torch triton` (Triton ships bundled with the Linux
  CUDA torch wheel since torch 2.0 — no separate install needed if
  `framework_native/.venv`-style torch is already present on the instance)
- Run: `python bench_triton.py` from this directory
