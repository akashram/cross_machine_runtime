# 2:4 Structured Sparsity

**Status: STUB — requires A100/H100. Run on p4d.24xlarge (A100).**

## What this measures
Prune a weight matrix to 2:4 structure (2 non-zeros per 4 elements), run sparse
matmul via `cusparseLtMatmul`, measure the 2x throughput vs dense baseline, and
document accuracy impact on a toy model.

## Implementation notes
- 2:4 constraint: in every group of 4 consecutive elements (within a row), exactly
  2 must be non-zero. A100+ hardware accelerates this specific pattern.
- Pruning: sort 4-element groups by absolute value, zero out the 2 smallest.
- cuSPARSELt API:
  - `cusparseLtInit`, `cusparseLtStructuredDescriptorInit`
  - `cusparseLtSpMMAPrune` to enforce 2:4 pattern
  - `cusparseLtMatmulDescriptorInit`, `cusparseLtMatmul`
- Expected: ~2x TFLOPS vs dense (hardware skips zero multiplies)
- Accuracy: measure perplexity / accuracy delta on a toy task after pruning

## Results

TODO: run on A100 hardware and fill in this table.

### Throughput (M=N=K=4096, FP16)

| Variant | TFLOPS | % of dense cuBLAS |
|---------|--------|-------------------|
| Dense cuBLAS | TODO | 100% |
| 2:4 sparse cusparseLt | TODO | TODO |

### Accuracy impact

| Model | Dense accuracy | 2:4 sparse accuracy | Delta |
|-------|---------------|---------------------|-------|
| Toy MLP (MNIST) | TODO | TODO | TODO |

## Hardware notes
- Required: A100 or H100 (Sparse Tensor Cores introduced in Ampere)
- Build preset: cuda (Linux) + cuSPARSELt library
- cuSPARSELt is separate from cuSPARSE — install separately
