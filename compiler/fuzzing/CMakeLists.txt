# Step 14: libFuzzer — requires MLIR + Clang with fuzzer support
add_executable(ir_fuzz ir_fuzz.cpp)
target_link_libraries(ir_fuzz PRIVATE
  RuntimeDialect
  ShapeInferencePass
  FusionPass
  AffinePass
  MemPlanningPass
  RematPass
  PlacementPass
  ShardingPass
  KernelSpecPass
  MLIRParser
  MLIRPass
)
target_include_directories(ir_fuzz PRIVATE
  ${CMAKE_SOURCE_DIR}/compiler/dialect
  ${CMAKE_SOURCE_DIR}/compiler/shape_inference
  ${CMAKE_SOURCE_DIR}/compiler/fusion
  ${CMAKE_SOURCE_DIR}/compiler/affine_lower
  ${CMAKE_SOURCE_DIR}/compiler/mem_planning
  ${CMAKE_SOURCE_DIR}/compiler/remat
  ${CMAKE_SOURCE_DIR}/compiler/placement
  ${CMAKE_SOURCE_DIR}/compiler/sharding
  ${CMAKE_SOURCE_DIR}/compiler/kernel_spec
)
target_compile_options(ir_fuzz PRIVATE -fsanitize=fuzzer,address)
target_link_options(ir_fuzz PRIVATE -fsanitize=fuzzer,address)
