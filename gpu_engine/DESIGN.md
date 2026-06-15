# GPU Backend Architecture

Status: Phase 3 in progress. Step 1 (CMake/CUDA integration) complete on paper;
remaining steps require a Linux GPU instance (Lambda Labs recommended — T4 for
steps 1–10, H100 for steps 16–19). This document covers design decisions made
*before* measurements are in hand; per-step READMEs will hold the numbers.

---

## 1. The one number that will explain most of Phase 3: the GPU ridge point

On an A100 (the target for serious work after the T4 learning phase):

| Resource | Peak |
|---|---|
| FP32 Tensor Core throughput | ~312 TFLOPS |
| HBM2e bandwidth | ~2 TB/s |
| Ridge point (FLOP/byte) | ~156 |

Every kernel in `gpu_engine/` will fall into one of two camps relative to
that ridge point — and the camp determines which optimizations can move the
needle:

| Camp | Kernels | Lever |
|---|---|---|
| Memory-bound (AI ≪ 156) | elementwise (add/mul/relu/gelu), softmax, layer norm | reduce HBM reads/writes (fuse ops, tiled SRAM) |
| Compute-bound (AI ≫ 156) | large GEMM (≥ 4096×4096), large convolutions | saturate Tensor Cores, hide latency with warp occupancy |

Flash Attention is the paradigmatic example of turning a memory-bound kernel
(naive attention: reads Q, K, V multiple times) into something closer to
compute-bound (tiled SRAM reuse of Q, K, V — 2-5× less HBM traffic at
long sequence lengths). Most of Phase 3 is about locating this boundary
per kernel and attacking the correct bottleneck.

The T4 has a different ridge point (~65 TFLOPS FP32, ~320 GB/s, ridge ≈ 203
FLOP/byte). Starting there keeps cloud costs low while the CMake/profiling
pipeline matures; move to A100/H100 for Flash Attention and Hopper-specific
steps.

---

## 2. Step 1 — CMake CUDA integration

### Decision: check_language / enable_language, not project(LANGUAGES CXX CUDA)

```cmake
# root CMakeLists.txt
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_subdirectory(gpu_engine)
endif()
```

**Alternative considered:** list CUDA directly in `project(LANGUAGES CXX CUDA)`.

**Why not:** on the development Mac (Apple clang 14, macOS Ventura) NVIDIA
dropped macOS CUDA support after CUDA 10.x — the CUDA toolkit simply does
not exist. A `project(LANGUAGES CXX CUDA)` that CMake cannot satisfy is a
hard configure error; the entire `cmake --preset debug` breaks. The
`check_language` pattern makes CUDA *optional*: the Mac continues to build
`foundation/` and `cpu_engine/` normally; `gpu_engine/` is silently skipped
with a status message. The Linux cloud instance has the toolkit and builds
everything.

### Decision: find_package(CUDAToolkit) inside gpu_engine for library targets

`enable_language(CUDA)` gives us `.cu` compilation. It does not automatically
expose `CUDA::cudart`, `CUDA::cublas`, `CUDA::cufft` etc. as CMake targets.
`find_package(CUDAToolkit REQUIRED)` inside `gpu_engine/CMakeLists.txt`
provides those imported targets.

We call `find_package` inside `gpu_engine/`, not at root level, so the
`REQUIRED` keyword only fires when we actually enter the gpu_engine subtree
(which only happens when `CMAKE_CUDA_COMPILER` is set — see above).

### Decision: CMAKE_CUDA_ARCHITECTURES

| Context | Setting | Reason |
|---|---|---|
| Development (cuda-debug / cuda-release preset) | `native` | CMake detects the GPU physically present in the machine; generates only one set of PTX/SASS — fastest compile |
| CI / broad compatibility | `"75;80;86;90"` | T4 (sm_75), A100 (sm_80), A10G (sm_86), H100 (sm_90) — all targets we will run on |

`native` is wrong for CI because the build machine may have a different GPU
than the test machine. Explicit arch lists in CI ensure the binary runs on
the target without JIT-recompile overhead.

### Decision: nvcc flags via cmake/Cuda.cmake helper

Rather than repeating the flag list in every subdirectory's CMakeLists, a
`target_apply_cuda_flags(target)` function in `cmake/Cuda.cmake` applies:

```
Release:  -O3 --use_fast_math -lineinfo --ptxas-options=-v
Debug:    -G -g -O0 -lineinfo --ptxas-options=-v
Both:     -Xcompiler=-Wall,-Wextra
```

**`--use_fast_math` in Release only:** enables `--ftz=true` (flush
denormals to zero), `--prec-div=false` (fast approximate reciprocal), and
`--prec-sqrt=false`. These break IEEE 754 strict compliance, which matters
for debugging but not for benchmarking throughput. **Never enable in Debug.**

**`-lineinfo` in both configs:** preserves source-to-PTX line mapping.
Nsight Compute uses this to annotate hotspots back to source lines. Adds
roughly 5% binary size, no runtime cost. Worth it always.

**`--ptxas-options=-v`:** prints register usage and spill counts per kernel
at compile time. A kernel spilling to local memory (register pressure →
spill to L1/global) will show in this output before you even profile it.
Make this visible at build time.

**`-G` (Debug only):** disables all compiler optimizations *including* the
optimizer inside ptxas. Required for accurate step-through debugging in
Nsight. **Never use -G when benchmarking** — it serializes warp execution
and produces throughput numbers that are off by 10–100×.

### Decision: separable compilation off by default

`CUDA_SEPARABLE_COMPILATION ON` lets device code in one `.cu` call
`__device__` functions defined in another `.cu`. It adds a link step (device
link) and increases compile time. We do not need cross-translation-unit
device calls for Phase 3 kernels (each kernel is self-contained in one
`.cu`). Turn it on only when a specific step requires it; document why.

---

## 3. Directory layout

```
gpu_engine/
  DESIGN.md                   ← this file
  CMakeLists.txt
  device_query/               ← Step 1: enumerate GPUs, verify setup
  memory/                     ← Step 2: cudaMalloc wrappers, tensor handle GPU extension
  streams/                    ← Step 3: stream pool, event sync, compute/transfer overlap
  warp_primitives/            ← Step 4: __shfl_sync, __ballot_sync, warp reduce/scan
  shared_mem/                 ← Step 5: bank-conflict-free reduction, prefix sum, transpose
  coalescing/                 ← Step 6: memory access pattern validator + Nsight integration
  occupancy/                  ← Step 7: cudaOccupancyMaxActiveBlocksPerMultiprocessor
  kernels/
    elementwise/              ← Step 8: add, mul, relu, gelu, softmax
    gemm/                     ← Step 9: naive → shared-mem tiled → WMMA → cuBLAS comparison
  ptx_sass/                   ← Step 10: cuobjdump workflow, SASS annotation examples
  flash_attn/                 ← Steps 11–12: forward + backward, tiled SRAM, online softmax
  graphs/                     ← Step 13: CUDA Graph capture + replay
  p2p/                        ← Step 14: GPUDirect P2P, peer cudaMemcpyAsync
  precision/                  ← Steps 15–17: BF16, FP8, Tensor Core alignment analysis
  hopper/                     ← Steps 18–19: TMA, WGMMA
  sparsity/                   ← Step 20: 2:4 structured sparsity, cusparseLtMatmul
  roofline/                   ← Step 21: peak FLOPS micro-benchmark, STREAM bandwidth
  mps/                        ← Step 22: MPS server setup, context-switch overhead
  power/                      ← Step 23: NVML power monitoring, thermal throttling
  ci/                         ← Step 24: ncu --set full in CI pipeline
  test/
  bench/
```

Mirrors the `cpu_engine/` pattern: one directory per step, own
CMakeLists.txt, README.md written after seeing numbers.

---

## 4. Tensor handle extension (Step 2 preview)

`foundation/tensor/tensor.h` already has `DeviceType::kCUDA = 1` stubbed.
Step 2 makes it real by adding a `TensorHandle::empty_cuda()` factory:

```cpp
// Allocate device memory; buffer_ destructor calls cudaFree.
static TensorHandle empty_cuda(std::span<const int64_t> shape,
                               Dtype dtype,
                               int device_index = 0) noexcept {
    cudaSetDevice(device_index);
    std::size_t bytes = static_cast<std::size_t>(numel_of(shape)) * dtype_size(dtype);
    void* raw = nullptr;
    if (cudaMalloc(&raw, bytes) != cudaSuccess) return {};
    auto buf = std::shared_ptr<void>(raw, [](void* p){ cudaFree(p); });
    Device dev{DeviceType::kCUDA, device_index};
    return TensorHandle(std::move(buf), raw,
                        to_vec(shape), contiguous_strides(shape, dtype),
                        dtype, dev);
}
```

The handle API (`shape`, `strides`, `dtype`, `device`, `data()`) is
unchanged. CPU code that holds a `TensorHandle` need not know whether the
buffer is on the device. The `device()` accessor tells callers which
memory space `data()` points into.

**Why shared_ptr<void> with cudaFree as deleter:** exactly the same
ownership model used for CPU (`std::free` as deleter). The ref-count
semantics (cheap handle copies, last-owner frees) apply equally to GPU
memory. The risk is calling `cudaFree` after the CUDA context is destroyed
(e.g., at static destruction time) — we will address this in Step 2 with
a context-lifetime guard.

---

## 5. Mac / Linux split

| Step | Builds on Mac | Requires Linux+GPU |
|---|---|---|
| 1 (CMake scaffold, device_query) | CMakeLists only | device_query.cu execution |
| 2–24 | — | All CUDA .cu compilation and execution |

The entire `gpu_engine/` subtree is gated behind `if(CMAKE_CUDA_COMPILER)` at
root level. `cmake --preset debug` on the Mac configures and builds
`foundation/` + `cpu_engine/` normally, skips `gpu_engine/` entirely. No
stubs, no `#ifdef __CUDACC__` gymnastics in the host-only tensor handle —
the CUDA extension lives exclusively in `gpu_engine/memory/` which is never
compiled on Mac.

---

## 6. Cloud instance progression

| Phase | Instance | GPU | Why |
|---|---|---|---|
| Steps 1–10 | Lambda g4dn.xlarge | T4 (sm_75, 16 GB GDDR6) | Cheapest, Turing Tensor Cores, Nsight works |
| Steps 11–15 | Lambda A10 or A100 | sm_86 or sm_80 | Flash Attention, BF16, larger memory |
| Steps 16–19 | Lambda H100 | sm_90 | TMA, WGMMA, FP8 — Hopper-only hardware |
| Steps 20–24 | A100 | sm_80 | cusparseLt 2:4 supported, roofline, MPS |

Terminate after each session. Lambda charges per minute. A T4 session
(compile + profile + terminate) costs roughly $1–2. An H100 session is ~$3–4.
