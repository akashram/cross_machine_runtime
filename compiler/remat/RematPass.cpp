//===- RematPass.cpp - Step 8 implementation ------------------------------===//

#include "RematPass.h"
#include "RuntimeDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace runtime {
namespace {

static bool isCheapToRecompute(Operation *op) {
  return isa<ReluOp, GeluOp, SigmoidOp, AddOp, MulOp, SubOp>(op);
}

struct RematPass : public PassWrapper<RematPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RematPass)

  RematPass() = default;
  explicit RematPass(double threshold) : threshold(threshold) {}

  StringRef getArgument() const final { return "runtime-remat"; }
  StringRef getDescription() const final {
    return "Clone cheap ops at distant use sites instead of keeping results resident";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Block &body = func.getBody().front();

    llvm::DenseMap<Operation *, int64_t> timeOf;
    int64_t t = 0;
    for (Operation &op : body) timeOf[&op] = t++;

    SmallVector<Operation *> candidates;
    func.walk([&](Operation *op) {
      if (isCheapToRecompute(op)) candidates.push_back(op);
    });

    int64_t opsInserted = 0;
    for (Operation *op : candidates) {
      Value result = op->getResult(0);
      SmallVector<Operation *> users(result.getUsers());
      if (users.size() < 2) continue; // nothing to shrink

      int64_t defTime = timeOf.lookup(op);
      int64_t farthest = defTime;
      for (Operation *u : users) farthest = std::max(farthest, timeOf.lookup(u));
      double span = static_cast<double>(farthest - defTime);
      if (span < threshold) continue; // resident cost doesn't justify recompute

      llvm::sort(users, [&](Operation *a, Operation *b) {
        return timeOf.lookup(a) < timeOf.lookup(b);
      });

      // Nearest use keeps the original result (shrinks its live range to
      // [defTime, timeOf(users[0])]); every farther use gets its own clone
      // planted immediately before it, so that clone's live range is ~0.
      for (Operation *user : llvm::ArrayRef(users).drop_front()) {
        OpBuilder builder(user);
        IRMapping mapping; // op's own operands are still valid SSA values at
                            // the clone site (they dominate `user` because
                            // they dominated `op`, which is earlier); only
                            // the *result* identity changes, not the inputs.
        Operation *clone = builder.clone(*op, mapping);
        for (OpOperand &operand : user->getOpOperands())
          if (operand.get() == result) operand.set(clone->getResult(0));
        ++opsInserted;
      }
    }

    func->setAttr("runtime.remat_ops_inserted",
                   Builder(&getContext()).getI64IntegerAttr(opsInserted));
  }

  double threshold = 8.0;
};

} // namespace

std::unique_ptr<Pass> createRematerializationPass(double remat_threshold) {
  return std::make_unique<RematPass>(remat_threshold);
}

static PassRegistration<RematPass> pass([] { return std::make_unique<RematPass>(8.0); });

} // namespace runtime
