#!/usr/bin/env bash
# inspect.sh — dump PTX and SASS for a compiled CUDA binary
# Usage: ./inspect.sh <cubin_or_fatbin> [kernel_name_filter]
#
# TODO: run on GPU hardware with CUDA toolkit installed

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> [kernel_filter]}"
FILTER="${2:-}"

echo "=== PTX ==="
# TODO: nvcc -ptx is done at compile time; add -ptx flag to nvcc in CMakeLists
echo "  Build with: nvcc -ptx ${BINARY%.cubin}.cu -o ${BINARY%.cubin}.ptx"

echo ""
echo "=== SASS ==="
if command -v cuobjdump &>/dev/null; then
    if [[ -n "${FILTER}" ]]; then
        cuobjdump --dump-sass "${BINARY}" | grep -A 50 "${FILTER}" || true
    else
        cuobjdump --dump-sass "${BINARY}"
    fi
else
    echo "  cuobjdump not found — install CUDA toolkit"
    echo "  Expected: cuobjdump --dump-sass ${BINARY}"
fi

echo ""
echo "=== Nsight source correlation ==="
echo "  ncu --source-level ptx --kernel-name ${FILTER:-<kernel>} ./${BINARY%.cubin}"
