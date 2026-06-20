#!/usr/bin/env bash
# ncu_profile.sh — run Nsight Compute full profiling on a benchmark binary
# Usage: ./ncu_profile.sh <binary> [output_prefix]
# TODO: run on Linux GPU hardware with ncu installed

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> [output_prefix]}"
PREFIX="${2:-$(basename "${BINARY}")}"
OUTDIR="benchmarks/nsight/$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

mkdir -p "${OUTDIR}"

CSV_FILE="${OUTDIR}/${PREFIX}_metrics.csv"
NCREP_FILE="${OUTDIR}/${PREFIX}.ncu-rep"

echo "[nsight_ci] Profiling: ${BINARY}"
echo "[nsight_ci] Output: ${CSV_FILE}"

# Full metric set — expensive, run only in CI not interactive dev
ncu \
    --set full \
    --csv \
    --log-file "${CSV_FILE}" \
    --export "${NCREP_FILE}" \
    "./${BINARY}"

echo "[nsight_ci] Profiling complete. Parse with: python3 parse_ncu.py ${CSV_FILE}"
