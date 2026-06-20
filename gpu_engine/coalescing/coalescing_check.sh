#!/usr/bin/env bash
# coalescing_check.sh — validate memory coalescing ratio for a CUDA kernel
# Usage: ./coalescing_check.sh <binary> <kernel_name> [threshold=0.90]
#
# TODO: run on GPU hardware with Nsight Compute installed

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> <kernel_name> [threshold]}"
KERNEL="${2:?Usage: $0 <binary> <kernel_name> [threshold]}"
THRESHOLD="${3:-0.90}"

METRICS="l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"

CSV_OUT=$(mktemp /tmp/ncu_coalesce_XXXXXX.csv)

# TODO: replace with real ncu invocation on GPU hardware
echo "[coalescing_check] TODO: run ncu on GPU hardware"
echo "  Command would be:"
echo "  ncu --kernel-name ${KERNEL} --metrics ${METRICS} --csv ${BINARY} > ${CSV_OUT}"

# TODO: parse CSV_OUT and compute:
#   ld_ratio  = sectors_ld / requests_ld
#   st_ratio  = sectors_st / requests_st
#   pass = (ld_ratio <= 1/threshold) && (st_ratio <= 1/threshold)
# Exit 1 if any kernel fails threshold.

echo "[coalescing_check] STUB: no GPU available, skipping validation"
exit 0
