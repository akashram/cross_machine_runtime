# Step 9: Device placement pass
add_mlir_library(PlacementPass
  PlacementPass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  CostModel
  MLIRPass
)

target_include_directories(PlacementPass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
  ${CMAKE_SOURCE_DIR}/compiler/cost_model
)
