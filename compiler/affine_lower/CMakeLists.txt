# Step 6: Affine dialect lowering + tiling
add_mlir_library(AffinePass
  AffinePass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  MLIRPass
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRMemRefDialect
)

target_include_directories(AffinePass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
)
