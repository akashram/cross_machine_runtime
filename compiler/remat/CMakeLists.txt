# Step 8: Rematerialization pass
add_mlir_library(RematPass
  RematPass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  MLIRPass
)

target_include_directories(RematPass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
)
