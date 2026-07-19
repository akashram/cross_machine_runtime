//===- AotCompiler.cpp - Step 12 implementation --------------------------===//
//
// Every stage's timer wraps exactly one PassManager.run() or one
// non-MLIR step (parse, LLVM translation, link) — see CompileStats in the
// header for what ends up in compiler/aot/README.md's results table.
//
//===----------------------------------------------------------------------===//

#include "AotCompiler.h"

#include "AffinePass.h"
#include "FusionPass.h"
#include "KernelSpecPass.h"
#include "MemPlanningPass.h"
#include "PlacementPass.h"
#include "RematPass.h"
#include "RuntimeDialect.h"
#include "ShapeInferencePass.h"
#include "ShardingPass.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include <chrono>
#include <cstdlib>

using namespace mlir;
using Clock = std::chrono::steady_clock;

namespace runtime {
namespace {

double elapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Runs one func::FuncOp-scoped pass over every function in `module`,
// returning elapsed ms. Kept as a free function because five of Phase 4's
// nine passes share this exact "new PassManager, one pass, time it" shape.
template <typename PassFactory>
double runFuncPass(ModuleOp module, PassFactory factory) {
  auto start = Clock::now();
  PassManager pm(module.getContext());
  pm.addNestedPass<func::FuncOp>(factory());
  (void)pm.run(module); // failures are non-fatal here: later stages just
                         // see whatever state the IR was left in, matching
                         // AotCompiler::compile()'s "log and continue"
                         // policy documented in the header.
  return elapsedMs(start);
}

double runModulePass(ModuleOp module, std::unique_ptr<Pass> pass) {
  auto start = Clock::now();
  PassManager pm(module.getContext());
  pm.addPass(std::move(pass));
  (void)pm.run(module);
  return elapsedMs(start);
}

} // namespace

AotCompiler::AotCompiler(CompilerOptions opts) : opts_(std::move(opts)) {}

bool AotCompiler::compile(const std::string &input_ir_path,
                           const std::string &output_binary_path) {
  auto totalStart = Clock::now();
  MLIRContext context;
  context.getOrLoadDialect<RuntimeDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();

  auto parseStart = Clock::now();
  OwningOpRef<ModuleOp> moduleRef = parseSourceFile<ModuleOp>(input_ir_path, &context);
  stats_.parse_ms = elapsedMs(parseStart);
  if (!moduleRef) return false;
  ModuleOp module = *moduleRef;

  stats_.shape_inference_ms = runFuncPass(module, &createShapeInferencePass);
  stats_.fusion_ms = runFuncPass(module, &createFusionPass);

  {
    auto start = Clock::now();
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(createAffineConversionPass());
    pm.addNestedPass<func::FuncOp>(createAffineTilingPass(opts_.affine_tile_size));
    (void)pm.run(module);
    stats_.affine_lower_ms = elapsedMs(start);
  }

  stats_.mem_planning_ms = runFuncPass(module, &createMemoryPlanningPass);
  // Remat and sharding are func::FuncOp-scoped passes (like the five
  // above), so they go through runFuncPass, not runModulePass — only
  // kernel_spec operates at module scope (it inserts symbol
  // declarations into the module body).
  stats_.remat_ms = runFuncPass(module, [&] {
    return createRematerializationPass(opts_.remat_threshold);
  });
  stats_.placement_ms = runFuncPass(module, &createPlacementPass);
  stats_.sharding_ms = runFuncPass(module, [&] {
    return createShardingPass(opts_.num_shard_devices,
                               static_cast<ShardingStrategy>(opts_.shard_strategy));
  });
  stats_.kernel_spec_ms = runModulePass(module, createKernelSpecializationPass());

  if (auto lowered = module->getAttrOfType<IntegerAttr>("runtime.kernels_lowered"))
    stats_.kernels_lowered = lowered.getInt();
  if (auto skipped = module->getAttrOfType<IntegerAttr>("runtime.kernels_skipped"))
    stats_.kernels_skipped = skipped.getInt();

  // Standard MLIR lowering to LLVM dialect: affine -> scf -> cf -> llvm,
  // plus arith/memref/func -> llvm, then reconcile the unrealized casts
  // those conversions leave behind at dialect boundaries.
  auto codegenStart = Clock::now();
  {
    PassManager pm(&context);
    pm.addPass(createLowerAffinePass());
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(pm.run(module))) {
      stats_.codegen_ms = elapsedMs(codegenStart);
      return false;
    }
  }

  llvm::LLVMContext llvmContext;
  registerBuiltinDialectTranslation(context);
  registerLLVMDialectTranslation(context);
  std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    stats_.codegen_ms = elapsedMs(codegenStart);
    return false;
  }

  std::string objPath = output_binary_path + ".o";
  {
    std::error_code ec;
    llvm::ToolOutputFile objFile(objPath, ec, llvm::sys::fs::OF_None);
    if (ec) { stats_.codegen_ms = elapsedMs(codegenStart); return false; }
    // Emitting via the target machine's addPassesToEmitFile is the
    // "real" path (see gpu_engine/ptx_sass for the analogous CUDA-side
    // object inspection); wiring a TargetMachine here is deferred to the
    // Linux build where an actual llc/TargetRegistry is available to
    // sanity-check against. For now, .ll text is emitted and clang
    // finishes IR->object->link in one shot below, which is a valid
    // (if less granular) AOT path and keeps this function buildable
    // against whatever LLVM version MLIR_DIR points at.
    llvmModule->print(objFile.os(), nullptr);
    objFile.keep();
  }

  std::string linkCmd = "clang -x ir \"" + objPath + "\" -o \"" + output_binary_path + "\"";
  for (const std::string &lib : opts_.link_libraries)
    linkCmd += " \"" + lib + "\"";
  if (opts_.verbose) linkCmd += " -v";
  int rc = std::system(linkCmd.c_str());
  stats_.codegen_ms = elapsedMs(codegenStart);
  stats_.total_ms = elapsedMs(totalStart);
  return rc == 0;
}

} // namespace runtime
