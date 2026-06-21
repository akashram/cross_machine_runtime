#!/usr/bin/env bash
# inspect.sh — dump and analyse PTX/SASS for a compiled CUDA binary.
#
# Usage:
#   ./inspect.sh <binary> [kernel_name_filter]
#
# Outputs (to stdout, pipe to a file for offline reading):
#   1. PTX from the embedded fatbinary (via cuobjdump --dump-ptx)
#   2. SASS disassembly (via cuobjdump --dump-sass)
#   3. SASS pattern summary (via sass_analyze.py)
#   4. Source-level PTX correlation command (ncu --source-level)
#
# Requires: CUDA toolkit (provides cuobjdump, nvdisasm).
#           ncu in PATH for source correlation.
#
# To keep PTX/SASS during compilation, build with:
#   -Xptxas -v            (register/smem usage per kernel)
#   --generate-line-info  (source correlation in ncu)
#   --keep                (preserves .ptx and .cubin intermediate files)
# The project cmake/Cuda.cmake already adds -Xptxas -v in release mode.
# Add --generate-line-info and --keep manually when doing a deep inspection run.

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> [kernel_filter]}"
FILTER="${2:-}"          # empty = all kernels

need_tool() {
    command -v "$1" &>/dev/null || { echo "ERROR: $1 not found in PATH"; exit 1; }
}
need_tool cuobjdump

SEP="$(printf '=%.0s' {1..70})"

# -----------------------------------------------------------------------
# 1. PTX
# -----------------------------------------------------------------------
echo "${SEP}"
echo "=== PTX (virtual ISA) ==="
echo "${SEP}"
if [[ -n "${FILTER}" ]]; then
    cuobjdump --dump-ptx "${BINARY}" | awk "
        /\.visible .entry ${FILTER}/, /^\}/" || true
else
    cuobjdump --dump-ptx "${BINARY}"
fi

# -----------------------------------------------------------------------
# 2. SASS
# -----------------------------------------------------------------------
echo ""
echo "${SEP}"
echo "=== SASS (machine ISA) ==="
echo "${SEP}"
if [[ -n "${FILTER}" ]]; then
    # Print the kernel header + its body (until the next blank line after last instr)
    cuobjdump --dump-sass "${BINARY}" | awk "
        /Function : .*${FILTER}/ { found=1 }
        found { print }
        found && /^\s*$/ && prev ~ /;/ { found=0 }
        { prev=\$0 }" || true
else
    cuobjdump --dump-sass "${BINARY}"
fi

# -----------------------------------------------------------------------
# 3. Register and shared-memory summary (-Xptxas -v output)
#    If the binary was compiled with -Xptxas -v, re-run nvcc isn't needed —
#    the info is already in the compilation log.  Here we use cuobjdump's
#    --list-kernels to show what's embedded.
# -----------------------------------------------------------------------
echo ""
echo "${SEP}"
echo "=== Embedded kernels ==="
echo "${SEP}"
cuobjdump --list-kernels "${BINARY}" || true

# -----------------------------------------------------------------------
# 4. SASS pattern analysis
# -----------------------------------------------------------------------
echo ""
echo "${SEP}"
echo "=== SASS pattern analysis ==="
echo "${SEP}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/sass_analyze.py" ]]; then
    SASS_TMP=$(mktemp /tmp/sass_XXXXXX.txt)
    trap "rm -f ${SASS_TMP}" EXIT
    cuobjdump --dump-sass "${BINARY}" > "${SASS_TMP}"
    python3 "${SCRIPT_DIR}/sass_analyze.py" "${SASS_TMP}" "${FILTER}"
else
    echo "sass_analyze.py not found — skipping pattern analysis"
fi

# -----------------------------------------------------------------------
# 5. Source-correlation command
# -----------------------------------------------------------------------
echo ""
echo "${SEP}"
echo "=== Source correlation (ncu) ==="
echo "${SEP}"
echo "Build with --generate-line-info, then run:"
echo ""
if [[ -n "${FILTER}" ]]; then
    echo "  ncu --source-level ptx --kernel-name ${FILTER} \\"
    echo "      --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,"
    echo "               sm__inst_executed.sum \\"
    echo "      ./${BINARY}"
else
    echo "  ncu --source-level ptx --set full ./${BINARY}"
fi
echo ""
echo "Tip: add '--export report.ncu-rep' and open in Nsight Compute GUI for"
echo "     the source → PTX → SASS three-panel view."
