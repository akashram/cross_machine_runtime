#!/usr/bin/env bash
# build_llvm.sh — build LLVM/MLIR from source
# Usage: ./build_llvm.sh <llvm-project-dir> <build-dir>
# TODO: run on Linux x86 instance

set -euo pipefail

LLVM_SRC="${1:?Usage: $0 <llvm-project-dir> <build-dir>}"
BUILD_DIR="${2:?Usage: $0 <llvm-project-dir> <build-dir>}"
JOBS="${3:-$(nproc)}"

mkdir -p "${BUILD_DIR}"

cmake -G Ninja \
    -S "${LLVM_SRC}/llvm" \
    -B "${BUILD_DIR}" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install"

cmake --build "${BUILD_DIR}" --target check-mlir -- -j"${JOBS}"

echo ""
echo "MLIR built successfully."
echo "Set MLIR_DIR=${BUILD_DIR}/lib/cmake/mlir when configuring this project."
