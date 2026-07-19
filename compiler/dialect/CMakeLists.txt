# Steps 2-3: Runtime dialect design + registration.
# Three separate mlir_tablegen() passes because LLVM_TARGET_DEFINITIONS is a
# single global var — one .td "entry point" per invocation, even though
# RuntimeOps.td transitively includes RuntimeDialect.td and RuntimeTypes.td.

set(LLVM_TARGET_DEFINITIONS RuntimeDialect.td)
mlir_tablegen(RuntimeDialect.h.inc -gen-dialect-decls -dialect=runtime)
mlir_tablegen(RuntimeDialect.cpp.inc -gen-dialect-defs -dialect=runtime)

set(LLVM_TARGET_DEFINITIONS RuntimeTypes.td)
mlir_tablegen(RuntimeEnums.h.inc -gen-enum-decls)
mlir_tablegen(RuntimeEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(RuntimeAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(RuntimeAttrDefs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS RuntimeOps.td)
mlir_tablegen(RuntimeOps.h.inc -gen-op-decls)
mlir_tablegen(RuntimeOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(RuntimeOpsIncGen)

add_mlir_dialect_library(RuntimeDialect
  RuntimeDialect.cpp

  DEPENDS
  RuntimeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
)

target_include_directories(RuntimeDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
