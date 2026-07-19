# Step 12: AOT compilation pipeline
add_mlir_library(AotCompiler
  AotCompiler.cpp

  LINK_LIBS PUBLIC
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
  MLIRTransforms
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
)

target_include_directories(AotCompiler PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
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

add_executable(runtime_aotc aotc_main.cpp)
target_link_libraries(runtime_aotc PRIVATE AotCompiler)
