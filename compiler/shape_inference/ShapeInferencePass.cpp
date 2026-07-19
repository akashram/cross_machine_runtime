//===- ShapeInferencePass.cpp - Step 4 implementation --------------------===//
//
// Worklist-based shape propagation, structurally the same algorithm as the
// MLIR "Toy" tutorial's ShapeInferencePass (mlir/examples/toy/Ch4): seed a
// worklist with every op that still has an unranked/dynamic result, and
// each time an op's inferred type is *more specific* than what it has now,
// commit the new type and re-enqueue that op's users. Terminates when
// either the worklist drains or a full pass produces no change (handles
// genuinely-dynamic dims — e.g. an input batch dim — without spinning).
//
//===----------------------------------------------------------------------===//

#include "ShapeInferencePass.h"
#include "RuntimeDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace runtime {
namespace {

// A type is "unresolved" if it isn't a ranked tensor with fully static
// dims. Elementwise/matmul results start unresolved whenever an operand is;
// propagation is done once every reachable op is resolved or a fixed point
// is reached with dynamic dims remaining (e.g. true dynamic batch size).
static bool isUnresolved(Type ty) {
  auto ranked = dyn_cast<RankedTensorType>(ty);
  return !ranked || !ranked.hasStaticShape();
}

static bool opHasUnresolvedResult(Operation *op) {
  return llvm::any_of(op->getResultTypes(), isUnresolved);
}

// True if `newTy` is at least as specific as `oldTy` (more static dims, or
// equal) — guards against thrashing between two equally-valid dynamic
// types, which would make the worklist never converge.
static bool isRefinement(Type oldTy, Type newTy) {
  auto oldRanked = dyn_cast<RankedTensorType>(oldTy);
  auto newRanked = dyn_cast<RankedTensorType>(newTy);
  if (!newRanked) return false;
  if (!oldRanked) return true;
  if (oldRanked.getRank() != newRanked.getRank()) return false;
  bool strictlyBetter = false;
  for (auto [o, n] : llvm::zip(oldRanked.getShape(), newRanked.getShape())) {
    if (o == n) continue;
    if (ShapedType::isDynamic(o) && !ShapedType::isDynamic(n)) { strictlyBetter = true; continue; }
    return false; // n regressed a previously-static dim, or conflicts
  }
  return strictlyBetter;
}

struct ShapeInferencePass
    : public PassWrapper<ShapeInferencePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  StringRef getArgument() const final { return "runtime-shape-inference"; }
  StringRef getDescription() const final {
    return "Propagate tensor shapes through runtime dialect ops to a fixed point";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::SetVector<Operation *> worklist;
    func.walk([&](Operation *op) {
      if (isa<InferTypeOpInterface>(op) && opHasUnresolvedResult(op))
        worklist.insert(op);
    });

    // Bound iterations at 4x op count: each op can only be refined finitely
    // many times (ranks are bounded), so this only trips on a genuine bug
    // (e.g. isRefinement not being a strict partial order) rather than on
    // legitimately-large graphs.
    const size_t maxIters = 4 * std::max<size_t>(1, worklist.size());
    size_t iters = 0;

    while (!worklist.empty()) {
      if (++iters > maxIters) {
        func.emitWarning("shape inference did not converge after ")
            << maxIters << " iterations; leaving remaining ops dynamic";
        break;
      }

      Operation *op = worklist.pop_back_val();
      auto infer = dyn_cast<InferTypeOpInterface>(op);
      if (!infer) continue;

      SmallVector<Type, 2> inferred;
      if (failed(infer.inferReturnTypes(
              op->getContext(), op->getLoc(), op->getOperands(),
              op->getDiscardableAttrDictionary(), op->getPropertiesStorage(),
              op->getRegions(), inferred)))
        continue; // not enough info yet (e.g. an operand still unranked)

      bool changed = false;
      for (auto [result, newTy] : llvm::zip(op->getResults(), inferred)) {
        if (!isRefinement(result.getType(), newTy)) continue;
        result.setType(newTy);
        changed = true;
        for (Operation *user : result.getUsers())
          if (isa<InferTypeOpInterface>(user))
            worklist.insert(user);
      }
      if (changed) {
        // fusion_group bodies carry their own internal InferTypeOpInterface
        // ops; re-run those too so a change to a group's input is seen by
        // ops inside the region before the group's own results are trusted.
        for (Region &region : op->getRegions())
          for (Operation &inner : region.getOps())
            if (isa<InferTypeOpInterface>(&inner))
              worklist.insert(&inner);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

static PassRegistration<ShapeInferencePass> pass;

} // namespace runtime
