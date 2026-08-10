# CUTLASS GEMM (Phase 3, Step 26)

**Status: STUB — requires CUDA GPU, compute capability 8.0+ (A100-class:
p3.2xlarge/p4d.24xlarge). Same instance class as `gpu_engine/kernels`'
WMMA path (step 9).** Added 2026-08-09 per the Phase 3 rescoping in
PLAN.md/CLAUDE.md.

## What this measures

A CUTLASS-template-instantiated GEMM (`cutlass::gemm::device::Gemm`, SM80
TensorOp, FP16 in / FP32 accumulate), benchmarked and correctness-checked
against `gpu_engine/kernels/gemm.cuh`'s hand-written `gemm_tiled`/`gemm_wmma`
(step 9) and, in written comparison only (no H100 here), against
`gpu_engine/hopper/wgmma.cuh`'s hand-written WGMMA kernel (step 19).

This is a *written comparison* deliverable per PLAN.md: where CUTLASS's
composable-template approach helps vs. where hand-written control still
wins.

## Files
- `cutlass_gemm.cu` — the GEMM instance + host driver (timing + correctness
  vs. CUTLASS's own host reference GEMM), same `cudaEvent_t` timing pattern
  as `kernels/gemm_bench.cu`
- `CMakeLists.txt` — `FetchContent`-pulls CUTLASS (header-only, v3.5.1,
  pinned tag) — this repo's first use of `FetchContent`; everything else
  external (gRPC/FlatBuffers/libfabric/libbpf) is a system package, see
  `networking/CMakeLists.txt`. Header-only + no distro package is exactly
  when `FetchContent` is the right tool instead of `find_package`.

## Where CUTLASS's template composition helps vs. where hand-written control wins

| Concern | Hand-written (`gemm.cuh`/`wgmma.cuh`) | CUTLASS (`cutlass_gemm.cu`) |
|---|---|---|
| Getting a *correct* Tensor Core kernel at all | `gemm_wmma` is ~50 lines of `wmma::` fragment calls the author must get exactly right (fragment shapes, leading dimensions, row/col-major agreement between load and store) | `Gemm` type alias — 8 template parameters, zero fragment-level code. This is the strongest case for CUTLASS: fragment-level MMA code is exactly the kind of finicky, hardware-generation-specific plumbing where a maintained library beats a hand roll. |
| Portability across SM generations | `gemm_wmma` targets Volta+ WMMA only; `wgmma.cuh` is a *separate*, hand-written PTX file for Hopper only, `#if __CUDA_ARCH__ >= 900`-guarded — two independent kernels to maintain | Swapping `ArchTag` (`Sm80` → `Sm90`) plus the collective-builder API (sketched, commented out, in `cutlass_gemm.cu`) is the intended CUTLASS-side path to the same WGMMA instructions `wgmma.cuh` hand-writes — one library, many backends, at the cost of the SM90 path being a materially different, newer API surface (`collective::CollectiveBuilder`) than the SM80 `device::Gemm` path, i.e. "swap one template parameter" undersells the real jump. |
| Pipelining / tile-size search | Not implemented in `gemm_tiled` (single-buffered); no search — `TILE` picked by hand, documented in `kernels/README.md` | `ThreadblockShape`/`WarpShape`/`InstructionShape`/pipeline-stage count are all template parameters CUTLASS ships well-tuned presets for per architecture — same tuning problem `gemm.py`'s `triton.autotune` configs solve by search, CUTLASS solves by shipping curated defaults instead |
| Debuggability when it's slow | `gemm_wmma` is ~50 lines; `cuobjdump --dump-sass` (step 10's workflow) on it is directly readable — the SASS maps back to source almost line-for-line | A `device::Gemm` instantiation expands into a deep template tree; `cuobjdump` still works but source-correlation is much harder — step 10's PTX/SASS inspection workflow is far more legible against `gemm.cuh`'s ~150 lines than against CUTLASS's generated code. This is the real cost of composition: when the *default* preset isn't fast enough for a specific shape, hand-written control (steps 9/19) is easier to actually go in and fix. |
| Non-standard fusion (e.g. GELU epilogue fused into the GEMM) | Would mean writing a new kernel variant by hand | CUTLASS's `epilogue::thread::LinearCombination` slot is designed to be swapped for a fused epilogue (e.g. `LinearCombinationRelu`) with no change to the mainloop — a real compositional win the hand-written kernels have no equivalent of (adding a fused epilogue to `gemm_wmma` means editing the kernel body) |

**The honest reading:** CUTLASS earns its complexity on the *mainloop* —
correct, portable Tensor Core MMA is exactly the kind of code where a
maintained template library should beat a from-scratch fragment-level
implementation, and steps 9/19 already show how much hand-written surface
area that mainloop is (a full WMMA kernel, then a *second* full kernel in a
different API for Hopper). It costs debuggability and predictability at the
edges — when a specific problem shape falls outside CUTLASS's well-tuned
presets, hand-written control (as in `gemm_tiled`/`gemm_wmma`) is the
easier thing to go in and fix, and `cuobjdump`-level inspection (step 10)
is far more tractable against 150 hand-written lines than against a
template-generated kernel.

## Results

TODO: run on GPU hardware and fill in this table.

### GEMM throughput (TFLOPS, FP16 in / FP32 accumulate, M=N=K)

| Size | CUTLASS SM80 TensorOp | gemm.cuh WMMA (kernels/README.md) | cuBLAS (kernels/README.md) |
|------|------------------------|---------------------------------------|----------------------------------|
| 512 | TODO | TODO | TODO |
| 1024 | TODO | TODO | TODO |
| 2048 | TODO | TODO | TODO |
| 4096 | TODO | TODO | TODO |

### Correctness (vs. CUTLASS host reference GEMM)

| Size | Result |
|------|--------|
| 512 | TODO |
| 1024 | TODO |
| 2048 | TODO |
| 4096 | TODO |

## Hardware notes
- Required: compute capability 8.0+ (A100-class) for the active `Sm80`
  instantiation; the commented-out `Sm90` collective-builder sketch would
  need an H100 (same gate as `gpu_engine/hopper`, never built here either)
- `FetchContent` pulls CUTLASS v3.5.1 headers at configure time — first
  build on a fresh instance needs network access to `github.com`
- Build preset: cuda (Linux), `-arch=sm_80`
