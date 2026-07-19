# Step 7: Memory planning pass
add_mlir_library(MemPlanningPass
  MemPlanningPass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  MLIRPass
)

target_include_directories(MemPlanningPass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
)
