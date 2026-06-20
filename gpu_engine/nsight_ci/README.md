# Nsight CI Integration

**Status: STUB — requires CUDA GPU + Nsight Compute CLI. Run on Linux GPU instance.**

## What this measures
Runs `ncu --set full` profiling on every benchmark binary in CI, parses the CSV
output to extract key metrics (SM utilization, memory bandwidth, warp efficiency),
and stores results per commit to track regressions.

## Implementation notes
- `ncu --set full --csv --log-file metrics.csv ./benchmark_binary`
- Key metrics to extract:
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed` (SM util %)
  - `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second` (HBM BW)
  - `smsp__thread_inst_executed_pred_on.avg.pct_of_peak_sustained_elapsed` (warp eff)
- Store as JSON per commit in `benchmarks/nsight/` (gitignored large files)
- CI check: alert if any metric regresses > 5% vs previous commit
- parse_ncu.py: reads CSV, extracts metrics, outputs structured JSON

## Results

TODO: run on GPU hardware and fill in this table.

| Kernel | SM util (%) | HBM BW (GB/s) | Warp eff (%) |
|--------|------------|---------------|--------------|
| gemm_wmma | TODO | TODO | TODO |
| flash_attn_fwd | TODO | TODO | TODO |
| elementwise_add | TODO | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU + Nsight Compute ≥ 2022.x (CLI: `ncu`)
- Linux only (ncu requires root or nvidia-smi group)
- Build preset: cuda release with -lineinfo
