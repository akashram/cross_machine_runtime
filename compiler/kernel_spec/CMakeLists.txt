# Step 11: Kernel specialization pass
add_mlir_library(KernelSpecPass
  KernelSpecPass.cpp

  LINK_LIBS PUBLIC
  RuntimeDialect
  MLIRPass
  MLIRFuncDialect
)

target_include_directories(KernelSpecPass PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/compiler/dialect
)
