# Phase 8 step 3: runtime dialect -> StableHLO lowering pass.
# Requires STABLEHLO_DIR pointing at an installed/built stablehlo checkout's
# cmake package dir (github.com/openxla/stablehlo, built against the same
# LLVM commit as MLIR_DIR — see compiler/mlir_setup/build_llvm.sh for the
# LLVM side of that pin).
find_package(Stablehlo REQUIRED CONFIG PATHS ${STABLEHLO_DIR} NO_DEFAULT_PATH)

add_mlir_library(StableHLOLowerPass
  StableHLOLowerPass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  StablehloOps
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  MLIRFuncDialect
)

target_include_directories(StableHLOLowerPass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
)
