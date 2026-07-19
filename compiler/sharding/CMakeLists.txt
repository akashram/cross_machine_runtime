# Step 10: Auto-sharding pass
add_mlir_library(ShardingPass
  ShardingPass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  MLIRPass
)

target_include_directories(ShardingPass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
)
