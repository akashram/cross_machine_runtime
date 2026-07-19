#!/usr/bin/env bash
# sweep_nccl_config.sh — sweep NCCL_ALGO/NCCL_PROTO/NCCL_BUFFSIZE against
# nccl-tests' all_reduce_perf, on an EFA-topology multi-GPU node, and
# record collective throughput before/after tuning. TODO: run on GPU
# nodes (Phase 3 gpu_engine/ hardware — p4d.24xlarge for real NVLink+EFA
# topology).
#
# Usage: ./sweep_nccl_config.sh <all_reduce_perf_path> <size_bytes>

set -euo pipefail
PERF_BIN="${1:?Usage: $0 <all_reduce_perf_path> <size_bytes>}"
SIZE="${2:?Usage: $0 <all_reduce_perf_path> <size_bytes>}"

ALGOS=(Ring Tree CollnetChain CollnetDirect NVLS)
PROTOS=(Simple LL LL128)
BUFFSIZES=(1048576 4194304 8388608)

echo "algo,proto,buffsize,busbw_gbs"
# Baseline: let NCCL auto-tune (its own internal cost model), for comparison.
env NCCL_DEBUG=WARN "$PERF_BIN" -b "$SIZE" -e "$SIZE" -g 1 \
    | awk '/algbw/{print "auto,auto,auto," $NF}' || true

for algo in "${ALGOS[@]}"; do
  for proto in "${PROTOS[@]}"; do
    for buffsize in "${BUFFSIZES[@]}"; do
      out=$(env NCCL_ALGO="$algo" NCCL_PROTO="$proto" NCCL_BUFFSIZE="$buffsize" \
                NCCL_DEBUG=WARN "$PERF_BIN" -b "$SIZE" -e "$SIZE" -g 1 2>/dev/null \
            | awk '/algbw/{print $NF}') || out="FAILED"
      echo "${algo},${proto},${buffsize},${out}"
    done
  done
done
