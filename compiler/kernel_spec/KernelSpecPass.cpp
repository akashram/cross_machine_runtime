//===- KernelSpecPass.cpp - Step 11 implementation ------------------------===//

#include "KernelSpecPass.h"
#include "RuntimeDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace runtime {
namespace {

// (op-kind, device) -> implementing symbol. TPU entries are forward
// references to Phase 8's lowering (StableHLO, not a direct kernel call —
// listed here for documentation completeness; the pass below skips TPU
// targets and leaves them for Phase 8's own lowering path).
StringRef lookupSymbol(Operation *op, DeviceKind device) {
  return llvm::TypeSwitch<Operation *, StringRef>(op)
      .Case<MatmulOp>([&](auto) -> StringRef {
        switch (device) {
          case DeviceKind::CPU:  return "cpu_engine::avx512::matvec_f32";
          case DeviceKind::GPU:  return "gpu_engine::kernels::gemm_wmma";
          case DeviceKind::FPGA: return "fpga_engine::dot_product";
          default: return "";
        }
      })
      .Case<ConvOp>([&](auto) -> StringRef {
        switch (device) {
          case DeviceKind::CPU:  return "cpu_engine::avx512::conv2d_f32";
          case DeviceKind::GPU:  return "gpu_engine::kernels::conv2d";
          default: return "";
        }
      })
      .Case<ReluOp>([&](auto) -> StringRef {
        switch (device) {
          case DeviceKind::CPU:  return "cpu_engine::branchless::relu_f32";
          case DeviceKind::GPU:  return "gpu_engine::kernels::relu";
          case DeviceKind::FPGA: return "fpga_engine::dot_product"; // README's documented example mapping
          default: return "";
        }
      })
      .Case<GeluOp>([&](auto) -> StringRef {
        return device == DeviceKind::GPU ? "gpu_engine::kernels::gelu" : "";
      })
      .Case<SigmoidOp>([&](auto) -> StringRef {
        return device == DeviceKind::GPU ? "gpu_engine::kernels::sigmoid" : "";
      })
      .Case<AddOp>([&](auto) -> StringRef {
        switch (device) {
          case DeviceKind::CPU: return "cpu_engine::avx512::add_f32";
          case DeviceKind::GPU: return "gpu_engine::kernels::elementwise_add";
          default: return "";
        }
      })
      .Case<SoftmaxOp>([&](auto) -> StringRef {
        return device == DeviceKind::GPU ? "gpu_engine::kernels::softmax" : "";
      })
      .Default([](Operation *) { return StringRef(); });
}

// fusion_kind + device -> fused kernel symbol. Only the pattern the
// fusion pass (step 5) actually emits (`{matmul,conv}[_bias]_{relu,gelu,sigmoid}`)
// needs entries; anything else falls through to "no match" and is left
// as an unlowered fusion_group.
StringRef lookupFusedSymbol(StringRef fusionKind, DeviceKind device) {
  if (device != DeviceKind::GPU) return ""; // fused kernels written for GPU first
  if (fusionKind == "matmul_bias_relu") return "gpu_engine::kernels::gemm_bias_relu";
  if (fusionKind == "matmul_bias_gelu") return "gpu_engine::kernels::gemm_bias_gelu";
  return "";
}

func::FuncOp getOrCreateDecl(ModuleOp module, StringRef name, FunctionType type) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name)) return existing;
  OpBuilder builder(module.getBodyRegion());
  auto decl = builder.create<func::FuncOp>(module.getLoc(), name, type);
  decl.setPrivate();
  return decl;
}

struct KernelSpecPass
    : public PassWrapper<KernelSpecPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KernelSpecPass)

  StringRef getArgument() const final { return "runtime-kernel-spec"; }
  StringRef getDescription() const final {
    return "Lower placed runtime dialect ops to runtime.kernel_call against real backend symbols";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    int64_t lowered = 0, skipped = 0;

    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "runtime" &&
          op->hasAttr(kDeviceAttrName) && !isa<KernelCallOp, TransferOp,
              AllGatherOp, ReduceScatterOp>(op))
        candidates.push_back(op);
    });

    for (Operation *op : candidates) {
      DeviceKind device = getAssignedDevice(op);
      OpBuilder builder(op);
      StringRef symbol;
      SmallVector<Value> callOperands(op->getOperands());

      if (auto group = dyn_cast<FusionGroupOp>(op)) {
        symbol = lookupFusedSymbol(group.getFusionKind(), device);
      } else {
        symbol = lookupSymbol(op, device);
      }

      if (symbol.empty()) { ++skipped; continue; }

      FunctionType fnType = builder.getFunctionType(
          ValueRange(callOperands).getTypes(), op->getResultTypes());
      getOrCreateDecl(module, symbol, fnType);

      auto call = builder.create<KernelCallOp>(
          op->getLoc(), op->getResultTypes(), callOperands,
          SymbolRefAttr::get(builder.getContext(), symbol),
          DeviceAttr::get(builder.getContext(), device));

      op->replaceAllUsesWith(call.getResults());
      op->erase();
      ++lowered;
    }

    module->setAttr("runtime.kernels_lowered", builder_i64(module, lowered));
    module->setAttr("runtime.kernels_skipped", builder_i64(module, skipped));
  }

  static IntegerAttr builder_i64(ModuleOp module, int64_t v) {
    return IntegerAttr::get(IntegerType::get(module.getContext(), 64), v);
  }
};

} // namespace

std::unique_ptr<Pass> createKernelSpecializationPass() {
  return std::make_unique<KernelSpecPass>();
}

static PassRegistration<KernelSpecPass> pass;

} // namespace runtime
