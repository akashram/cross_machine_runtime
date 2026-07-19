# Step 4: Shape inference pass
add_mlir_library(ShapeInferencePass
  ShapeInferencePass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  MLIRPass
  MLIRInferTypeOpInterface
)

target_include_directories(ShapeInferencePass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
)
