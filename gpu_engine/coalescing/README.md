# Memory Coalescing Validator

**Status: STUB — requires CUDA GPU + Nsight Compute CLI. Run on g4dn.xlarge or better.**

## What this measures
Validates that all GPU kernels achieve ≥90% coalesced global memory access by
running `ncu` with memory sector/request metrics and computing the coalescing ratio.

## Implementation notes
- Metric pair: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`
- Ideal ratio = 1.0 (one 128-byte sector per request); ratio > ~1.1 indicates uncoalesced access
- Integrate into CI: fail the build if any benchmark kernel falls below 0.90 threshold
- Also measure store coalescing: `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum`

## Results

TODO: run on GPU hardware and fill in this table.

| Kernel | Sectors/Request (load) | Sectors/Request (store) | Pass (≥90%)? |
|--------|------------------------|-------------------------|--------------|
| add_kernel | TODO | TODO | TODO |
| matmul_naive | TODO | TODO | TODO |
| matmul_tiled | TODO | TODO | TODO |
| flash_attn_fwd | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU (Volta+) + Nsight Compute ≥ 2022.x
- Build preset: cuda (Linux)
- Run: `./coalescing_check.sh <kernel_binary> <kernel_name>`
