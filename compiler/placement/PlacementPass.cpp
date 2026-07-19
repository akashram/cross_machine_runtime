//===- PlacementPass.cpp - Step 9 implementation --------------------------===//

#include "PlacementPass.h"
#include "CostModel.h"
#include "RuntimeDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace runtime {
namespace {

constexpr std::array<DeviceType, 4> kCandidates = {
    DeviceType::CPU, DeviceType::GPU, DeviceType::FPGA, DeviceType::TPU};

DeviceKind toDeviceKind(DeviceType d) {
  switch (d) {
    case DeviceType::CPU: return DeviceKind::CPU;
    case DeviceType::GPU: return DeviceKind::GPU;
    case DeviceType::FPGA: return DeviceKind::FPGA;
    case DeviceType::TPU: return DeviceKind::TPU;
  }
  return DeviceKind::Unassigned;
}

// Best-effort Shape extraction: rank-2 tensors map cleanly; anything else
// is flattened to a 2D [rows, cols] approximation. Good enough for a
// greedy cost comparison across devices — the ratio between devices'
// estimates is what drives the decision, not the absolute number.
Shape shapeOf(RankedTensorType ty) {
  if (!ty || !ty.hasStaticShape() || ty.getRank() == 0) return {1, 1, 1, 1};
  if (ty.getRank() == 1) return {1, ty.getDimSize(0), 1, 1};
  int64_t rows = ty.getDimSize(ty.getRank() - 2);
  int64_t cols = ty.getDimSize(ty.getRank() - 1);
  int64_t batch = 1;
  for (int i = 0; i < ty.getRank() - 2; ++i) batch *= ty.getDimSize(i);
  return {batch, rows, cols, cols};
}

// Returns {op-type, representative shape} for cost estimation, or
// std::nullopt for ops this pass doesn't place (transfer/collectives/
// kernel_call — already device-bound by construction).
std::optional<std::pair<OpType, Shape>> costKeyFor(Operation *op) {
  if (auto mm = dyn_cast<MatmulOp>(op)) {
    auto lhsTy = dyn_cast<RankedTensorType>(mm.getLhs().getType());
    auto rhsTy = dyn_cast<RankedTensorType>(mm.getRhs().getType());
    if (!lhsTy || !rhsTy || !lhsTy.hasStaticShape() || !rhsTy.hasStaticShape())
      return std::nullopt;
    int64_t rank = lhsTy.getRank();
    int64_t batch = 1;
    for (int i = 0; i < rank - 2; ++i) batch *= lhsTy.getDimSize(i);
    return std::make_pair(OpType::Matmul,
        Shape{batch, lhsTy.getDimSize(rank - 2), rhsTy.getDimSize(rhsTy.getRank() - 1),
              lhsTy.getDimSize(rank - 1)});
  }
  if (isa<ConvOp>(op)) {
    // im2col-equivalent packing (see CostModel.h): approximate from the
    // op's own result/operand shapes rather than threading kH/kW through
    // here — good enough for a greedy device comparison.
    auto conv = cast<ConvOp>(op);
    auto resTy = dyn_cast<RankedTensorType>(conv.getResult().getType());
    return std::make_pair(OpType::Conv, shapeOf(resTy));
  }
  if (isa<AddOp, MulOp, SubOp, ReluOp, GeluOp, SigmoidOp, SoftmaxOp>(op)) {
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    return std::make_pair(OpType::Elementwise, shapeOf(resTy));
  }
  if (isa<ReduceOp>(op)) {
    auto in = op->getOperand(0);
    return std::make_pair(OpType::Reduce, shapeOf(dyn_cast<RankedTensorType>(in.getType())));
  }
  if (auto group = dyn_cast<FusionGroupOp>(op)) {
    // Dominated by the matmul/conv the fusion pass (step 5) built the
    // group around — approximate with the first two inputs as a matmul.
    // Refined once kernel_spec (step 11) has real per-fused-kernel cost.
    if (group.getInputs().size() >= 2) {
      auto aTy = dyn_cast<RankedTensorType>(group.getInputs()[0].getType());
      auto bTy = dyn_cast<RankedTensorType>(group.getInputs()[1].getType());
      if (aTy && bTy && aTy.hasStaticShape() && bTy.hasStaticShape())
        return std::make_pair(OpType::Matmul,
            Shape{1, aTy.getDimSize(aTy.getRank() - 2),
                  bTy.getDimSize(bTy.getRank() - 1), aTy.getDimSize(aTy.getRank() - 1)});
    }
    return std::nullopt;
  }
  return std::nullopt;
}

struct PlacementPass : public PassWrapper<PlacementPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlacementPass)

  StringRef getArgument() const final { return "runtime-placement"; }
  StringRef getDescription() const final {
    return "Greedily assign runtime.device per op, minimizing compute + transfer cost";
  }

  // Function arguments are assumed host(CPU)-resident by convention —
  // the common case for this project (inputs arrive over the network or
  // from disk on the CPU-attached host before being dispatched to an
  // accelerator).
  DeviceKind deviceOf(Value v, llvm::DenseMap<Operation *, DeviceKind> &assigned) {
    if (Operation *def = v.getDefiningOp())
      return assigned.lookup(def);
    return DeviceKind::CPU;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    llvm::DenseMap<Operation *, DeviceKind> assigned;

    SmallVector<Operation *> ops;
    func.walk([&](Operation *op) { ops.push_back(op); });

    for (Operation *op : ops) {
      auto key = costKeyFor(op);
      if (!key) continue;
      auto [opType, shape] = *key;

      DeviceType best = DeviceType::CPU;
      double bestCost = std::numeric_limits<double>::infinity();
      for (DeviceType candidate : kCandidates) {
        DeviceCost dc = get_device_cost(candidate);
        double cost = estimate_us(candidate, opType, shape, dc);
        for (Value operand : op->getOperands()) {
          DeviceKind srcKind = deviceOf(operand, assigned);
          DeviceType src = (srcKind == DeviceKind::GPU) ? DeviceType::GPU
                          : (srcKind == DeviceKind::FPGA) ? DeviceType::FPGA
                          : (srcKind == DeviceKind::TPU) ? DeviceType::TPU
                          : DeviceType::CPU;
          if (src == candidate) continue;
          auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
          cost += estimate_us(candidate, OpType::Transfer, shapeOf(operandTy), dc);
        }
        if (cost < bestCost) { bestCost = cost; best = candidate; }
      }

      DeviceKind chosen = toDeviceKind(best);
      setAssignedDevice(op, chosen);
      assigned[op] = chosen;

      // Make the placement concrete: insert transfers for any operand
      // crossing a device boundary, so kernel specialization (step 11)
      // never has to re-derive it.
      OpBuilder builder(op);
      for (OpOperand &operand : op->getOpOperands()) {
        DeviceKind srcKind = deviceOf(operand.get(), assigned);
        if (srcKind == chosen) continue;
        auto transfer = builder.create<TransferOp>(
            op->getLoc(), operand.get().getType(), operand.get(),
            DeviceAttr::get(&getContext(), srcKind),
            DeviceAttr::get(&getContext(), chosen));
        operand.set(transfer.getResult());
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPlacementPass() { return std::make_unique<PlacementPass>(); }

static PassRegistration<PlacementPass> pass;

} // namespace runtime
