# GPUDirect Storage

**Status: code-complete, hardware-gated — real cuFile API usage, matching
Phase 3's convention for GPU code that hasn't run yet. Unlike step 1, there
is no portable CPU-only subset (see Design below).**

## What this measures

PLAN.md Phase 6 step 2: direct NVMe -> GPU checkpoint loading via GDS
(`cuFileRead`), measured against a CPU-staged baseline (`pread` into pinned
host memory, then `cudaMemcpy` H2D) — the path every checkpoint load takes
without GDS.

## Design

- `gds_reader.h`: `GdsFileReader` (open + `O_DIRECT` + `cuFileHandleRegister`,
  `cuFileRead` straight into a `cudaMalloc`'d device pointer) and
  `GdsDriverGuard` (RAII `cuFileDriverOpen`/`Close`, must bracket all
  `GdsFileReader` usage in the process).
- `gds_bench.cu`: writes a 2GB scratch file (representative single-shard
  checkpoint size), times both paths reading it fully into GPU memory.
- **Why this step has no portable local component, unlike step 1's data
  loader**: GDS's entire value proposition is the DMA engine on the NVMe
  controller writing straight into GPU HBM over PCIe, bypassing every CPU
  core and the page cache entirely. There's nothing left to simulate on a
  CPU-only Mac — a "GDS without a GPU or GDS-capable NVMe" isn't a smaller
  version of the same thing, it's a different thing (the staged-copy
  baseline it's supposed to beat). Contrast with step 1, where the
  algorithm (sharding, prefetching) is meaningful independent of what
  hardware eventually consumes the samples.

## Results
TODO: run on GPU hardware with a GDS-capable NVMe device (`p4d.24xlarge`
qualifies; verify with `gdscheck.py -p` from the GDS install first — some
instance/AMI combinations only support compatibility mode, which still
uses the cuFile API but falls back to a kernel bounce buffer internally).

| Checkpoint size | Staged (pread+memcpy) | GDS (cuFileRead) | Speedup |
|-----------------|------------------------|-------------------|---------|
| 2 GB | TODO | TODO | TODO |

## Hardware notes
- Required: CUDA toolkit with the GDS component, `libcufile`, a
  GDS-capable NVMe device (local NVMe on `p4d.24xlarge`, not network-
  attached EBS).
- Falls back gracefully at configure time: `CMakeLists.txt` skips this
  target entirely (not a build error) when `libcufile`/`cufile.h` aren't
  found, same pattern as the CUDA-toolkit gate in the root `CMakeLists.txt`.
