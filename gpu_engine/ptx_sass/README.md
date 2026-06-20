# PTX/SASS Inspection

**Status: STUB — requires CUDA GPU + cuobjdump. Run on any CUDA Linux instance.**

## What this measures
Inspects compiler-generated PTX and SASS assembly for key kernels. Identifies
inefficiencies: unnecessary memory transactions, suboptimal instruction scheduling,
spilled registers, missed vectorization. Documents findings per kernel.

## Implementation notes
- PTX (virtual ISA): `nvcc -ptx kernel.cu -o kernel.ptx`
- SASS (real ISA): `cuobjdump --dump-sass kernel.cubin`
- Key things to look for in SASS:
  - `LDG.E` vs `LDG.E.128` (vectorized vs scalar global loads)
  - `LDSM` (shared memory matrix load for Tensor Cores)
  - `STL` vs `STG` (local/spill store vs global store) — spills hurt occupancy
  - `DEPBAR` / `__syncthreads` cost
  - Dual-issue slots: SASS lines starting with `-` are co-issued
- Use `ncu --source-level ptx` for source correlation

## Results

TODO: run on GPU hardware and fill in this table.

| Kernel | Top inefficiency found | Fix applied | Speedup after fix |
|--------|----------------------|-------------|-------------------|
| gemm_naive | TODO | TODO | TODO |
| gemm_tiled | TODO | TODO | TODO |
| elementwise_add | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU + CUDA toolkit (cuobjdump included)
- Build preset: cuda release (Linux) — need -lineinfo for source correlation
- Command: `./inspect.sh <kernel.cubin> > analysis.txt`
