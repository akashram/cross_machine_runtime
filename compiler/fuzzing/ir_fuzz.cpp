// ir_fuzz.cpp — libFuzzer entry point for the runtime dialect IR parser
// and the full pass pipeline. Parses the fuzz input as MLIR text; on a
// successful parse, runs every Phase 4 pass over it in pipeline order.
// Parse failures on malformed input are the expected common case and are
// not bugs — libFuzzer is here to catch crashes/UB/assertion failures in
// the parser or in a pass that assumes shape-inference-clean IR and gets
// handed something it isn't (e.g. a pass indexing a rank-2 assumption
// against a rank-0 tensor the fuzzer generated).

#include "AffinePass.h"
#include "FusionPass.h"
#include "KernelSpecPass.h"
#include "MemPlanningPass.h"
#include "PlacementPass.h"
#include "RematPass.h"
#include "RuntimeDialect.h"
#include "ShapeInferencePass.h"
#include "ShardingPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  std::string irText(reinterpret_cast<const char *>(data), size);

  MLIRContext ctx;
  ctx.getOrLoadDialect<runtime::RuntimeDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<affine::AffineDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
  ctx.getOrLoadDialect<bufferization::BufferizationDialect>();
  // Malformed/incomplete IR is the overwhelmingly common case for random
  // input — a diagnostic handler that swallows output keeps the fuzzer's
  // stderr usable instead of drowning real crash reports in parse noise.
  ctx.getDiagEngine().registerHandler([](Diagnostic &) { return success(); });

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(irText, &ctx);
  if (!module) return 0;

  PassManager pm(&ctx);
  pm.addNestedPass<func::FuncOp>(runtime::createShapeInferencePass());
  pm.addNestedPass<func::FuncOp>(runtime::createFusionPass());
  pm.addNestedPass<func::FuncOp>(runtime::createAffineConversionPass());
  pm.addNestedPass<func::FuncOp>(runtime::createAffineTilingPass(32));
  pm.addNestedPass<func::FuncOp>(runtime::createMemoryPlanningPass());
  pm.addNestedPass<func::FuncOp>(runtime::createRematerializationPass(8.0));
  pm.addNestedPass<func::FuncOp>(runtime::createPlacementPass());
  pm.addNestedPass<func::FuncOp>(
      runtime::createShardingPass(4, runtime::ShardingStrategy::DataParallel));
  pm.addPass(runtime::createKernelSpecializationPass());
  (void)pm.run(*module); // ignore failures — we're fuzzing for crashes, not correctness

  return 0;
}
