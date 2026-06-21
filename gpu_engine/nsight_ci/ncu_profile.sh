#!/usr/bin/env bash
# ncu_profile.sh — run Nsight Compute full profiling on one or more benchmark binaries.
#
# Usage:
#   ./ncu_profile.sh <binary> [output_prefix] [--baseline <baseline.json>]
#
# Examples:
#   ./ncu_profile.sh ./build/release/roofline_bench
#   ./ncu_profile.sh ./build/release/gemm_bench gemm --baseline baselines/gemm.json
#
# Output (per commit):
#   benchmarks/nsight/<git-sha>/<prefix>_metrics.csv   — raw ncu CSV
#   benchmarks/nsight/<git-sha>/<prefix>_metrics.json  — parsed JSON
#   benchmarks/nsight/<git-sha>/<prefix>.ncu-rep       — Nsight UI report
#
# Requires:
#   - ncu in PATH (part of CUDA Toolkit, usually /usr/local/cuda/bin/ncu)
#   - python3 (for parse_ncu.py)
#   - Root or ncu permission (cudaLimitDevRuntimeSyncDepth): sudo ncu or
#     set /proc/driver/nvidia/params NVreg_RestrictProfilingToAdminUsers=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BINARY="${1:?Usage: $0 <binary> [output_prefix] [--baseline <path>]}"
PREFIX="${2:-$(basename "${BINARY}")}"
BASELINE=""

# Parse optional --baseline flag
shift 2 2>/dev/null || shift $# 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline) BASELINE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
OUTDIR="benchmarks/nsight/${GIT_SHA}"
mkdir -p "${OUTDIR}"

CSV_FILE="${OUTDIR}/${PREFIX}_metrics.csv"
JSON_FILE="${OUTDIR}/${PREFIX}_metrics.json"
REP_FILE="${OUTDIR}/${PREFIX}.ncu-rep"

echo "[nsight_ci] Binary   : ${BINARY}"
echo "[nsight_ci] Output   : ${OUTDIR}/"
echo "[nsight_ci] Git SHA  : ${GIT_SHA}"
echo ""

# Verify ncu is available
if ! command -v ncu &>/dev/null; then
    echo "[nsight_ci] ERROR: ncu not found in PATH." >&2
    echo "  Add /usr/local/cuda/bin to PATH or install CUDA Toolkit." >&2
    exit 1
fi

# Profile — --set full captures all hardware metrics (slow, ~5× overhead).
# Use --set basic for quick CI checks, full for detailed debugging.
echo "[nsight_ci] Running ncu --set full ..."
ncu \
    --set full \
    --csv \
    --log-file "${CSV_FILE}" \
    --export "${REP_FILE}" \
    --target-processes all \
    "${BINARY}"

echo "[nsight_ci] CSV written to: ${CSV_FILE}"

# Parse CSV → JSON with summary table
echo "[nsight_ci] Parsing metrics..."
PARSE_ARGS="--summary"
if [[ -n "${BASELINE}" ]]; then
    PARSE_ARGS="${PARSE_ARGS} --baseline ${BASELINE}"
fi

python3 "${SCRIPT_DIR}/parse_ncu.py" \
    "${CSV_FILE}" \
    --output "${JSON_FILE}" \
    ${PARSE_ARGS}

echo "[nsight_ci] JSON written to: ${JSON_FILE}"
echo ""
echo "To view in Nsight UI:"
echo "  ncu-ui ${REP_FILE}"
echo ""
echo "To check regressions vs baseline:"
echo "  python3 ${SCRIPT_DIR}/parse_ncu.py ${CSV_FILE} --baseline <prev_metrics.json>"
echo ""
echo "To update baseline after performance improvements:"
echo "  cp ${JSON_FILE} ${SCRIPT_DIR}/baselines/${PREFIX}.json"
