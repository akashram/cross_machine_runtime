# Step 5: Operator fusion pass
add_mlir_library(FusionPass
  FusionPass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)

target_include_directories(FusionPass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
)
