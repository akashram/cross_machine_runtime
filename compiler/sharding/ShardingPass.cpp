//===- ShardingPass.cpp - Step 10 implementation --------------------------===//
//
// Each strategy below mirrors a specific Megatron-LM pattern (see Phase 6,
// PLAN.md steps 11-13): column-parallel and row-parallel linear layers are
// the two building blocks that compose into tensor-parallel attention and
// MLP blocks. This pass validates the IR-level mechanics of that
// composition — shard-spec propagation and collective insertion — using
// the `runtime.shard_spec` attribute and `all_gather`/`reduce_scatter` ops
// already defined in the dialect (steps 2-3).
//
//===----------------------------------------------------------------------===//

#include "ShardingPass.h"
#include "RuntimeDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace runtime {
namespace {

ShardSpecAttr replicated(MLIRContext *ctx, int64_t rank, int64_t numDevices) {
  SmallVector<int64_t> partitions(rank, -1);
  return ShardSpecAttr::get(ctx, numDevices, partitions);
}

ShardSpecAttr shardedAt(MLIRContext *ctx, int64_t rank, int64_t dim, int64_t numDevices) {
  SmallVector<int64_t> partitions(rank, -1);
  if (dim >= 0 && dim < rank) partitions[dim] = numDevices;
  return ShardSpecAttr::get(ctx, numDevices, partitions);
}

struct ShardingPass : public PassWrapper<ShardingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardingPass)

  ShardingPass() = default;
  ShardingPass(int numDevices, ShardingStrategy strategy)
      : numDevices(numDevices), strategy(strategy) {}

  StringRef getArgument() const final { return "runtime-sharding"; }
  StringRef getDescription() const final {
    return "Apply a canned GSPMD-style sharding strategy to runtime.matmul ops";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    SmallVector<MatmulOp> matmuls;
    func.walk([&](MatmulOp op) { matmuls.push_back(op); });

    for (MatmulOp op : matmuls) {
      auto lhsTy = dyn_cast<RankedTensorType>(op.getLhs().getType());
      auto rhsTy = dyn_cast<RankedTensorType>(op.getRhs().getType());
      auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());
      if (!lhsTy || !rhsTy || !resTy) continue;

      switch (strategy) {
      case ShardingStrategy::Replicated: {
        op->setAttr(kShardAttrName, replicated(ctx, resTy.getRank(), numDevices));
        break;
      }
      case ShardingStrategy::DataParallel: {
        // Shard the leading (batch/token) dim of lhs and of the result;
        // the weight (rhs) is replicated on every device. No collective
        // needed within the forward matmul itself — gradient all-reduce
        // for data parallelism happens in the training loop (Phase 6),
        // out of scope for this dialect-level pass.
        op->setAttr(kShardAttrName, shardedAt(ctx, resTy.getRank(), 0, numDevices));
        break;
      }
      case ShardingStrategy::TensorParallelColumn: {
        // rhs sharded on its output-feature (last) dim; lhs replicated.
        // Result is naturally column-sharded — leave it that way (a
        // consumer that needs a replicated view inserts its own
        // all_gather; forcing one here would defeat the point of
        // column-parallel, which is to feed a row-parallel layer next
        // without ever materializing the full activation).
        op->setAttr(kShardAttrName,
                     shardedAt(ctx, resTy.getRank(), resTy.getRank() - 1, numDevices));
        break;
      }
      case ShardingStrategy::TensorParallelRow: {
        // Contraction dim (lhs last, rhs first) sharded on both operands:
        // each device holds a partial sum over its K-shard. The true
        // output requires summing those partial results across devices —
        // modeled here as reduce_scatter (all-reduce + redistribute),
        // since a distinct "partial-sum" tensor state isn't part of the
        // dialect's type system (documented simplification vs. real
        // GSPMD/XLA SPMD partitioner, which does model it explicitly).
        OpBuilder builder(op->getBlock(), std::next(Block::iterator(op)));
        auto reduced = builder.create<ReduceScatterOp>(
            op.getLoc(), resTy, op.getResult(), /*shard_dim=*/0, numDevices);
        op.getResult().replaceAllUsesExcept(reduced.getResult(), reduced);
        reduced->setAttr(kShardAttrName, shardedAt(ctx, resTy.getRank(), 0, numDevices));
        break;
      }
      }
    }
  }

  int numDevices = 1;
  ShardingStrategy strategy = ShardingStrategy::Replicated;
};

} // namespace

std::unique_ptr<Pass> createShardingPass(int num_devices, ShardingStrategy strategy) {
  return std::make_unique<ShardingPass>(num_devices, strategy);
}

static PassRegistration<ShardingPass> pass(
    [] { return std::make_unique<ShardingPass>(1, ShardingStrategy::Replicated); });

} // namespace runtime
