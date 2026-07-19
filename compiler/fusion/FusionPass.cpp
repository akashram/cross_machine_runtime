//===- FusionPass.cpp - Step 5 implementation ----------------------------===//
//
// Reduces HBM round-trips for the extremely common {matmul,conv}+bias+act
// shape: unfused, the bias-add and activation each read/write the full
// output tensor from HBM once more (2 extra read + 2 extra write passes
// over an M*N tensor). Fused into one `runtime.fusion_group`, kernel
// specialization (step 11) is free to emit a single kernel that keeps the
// intermediate in registers/shared memory — this pass only does the IR
// grouping; the memory-traffic win is realized in step 11's codegen.
//
//===----------------------------------------------------------------------===//

#include "FusionPass.h"
#include "RuntimeDialect.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace runtime {
namespace {

// Chain producing `act`'s input, matched back through an optional bias-add
// to a matmul or conv. Every op in the chain must have exactly one use
// (itself) or we'd be duplicating compute by fusing.
struct MatchedChain {
  Operation *root = nullptr;   // MatmulOp or ConvOp
  AddOp bias = nullptr;        // null if no bias-add in the chain
  StringRef kind;               // e.g. "matmul_bias_relu", "conv_relu"
};

static bool singleUse(Operation *op) { return op && op->hasOneUse(); }

template <typename ActOp>
static FailureOr<MatchedChain> matchChain(ActOp actOp, StringRef actName) {
  Value in = actOp.getInput();
  Operation *producer = in.getDefiningOp();
  if (!producer) return failure();

  MatchedChain chain;
  if (auto add = dyn_cast<AddOp>(producer)) {
    if (!singleUse(add)) return failure();
    Operation *lhsDef = add.getLhs().getDefiningOp();
    Operation *rhsDef = add.getRhs().getDefiningOp();
    Operation *matmulLike = nullptr;
    if (isa_and_nonnull<MatmulOp, ConvOp>(lhsDef)) matmulLike = lhsDef;
    else if (isa_and_nonnull<MatmulOp, ConvOp>(rhsDef)) matmulLike = rhsDef;
    if (!matmulLike || !singleUse(matmulLike)) return failure();
    chain.root = matmulLike;
    chain.bias = add;
  } else if (isa<MatmulOp, ConvOp>(producer)) {
    if (!singleUse(producer)) return failure();
    chain.root = producer;
  } else {
    return failure();
  }

  bool isMatmul = isa<MatmulOp>(chain.root);
  static llvm::SmallDenseMap<std::pair<bool, bool>, StringRef> kindNames = {
      {{true, true}, "matmul_bias_"}, {{true, false}, "matmul_"},
      {{false, true}, "conv_bias_"}, {{false, false}, "conv_"}};
  static std::string scratch;
  scratch = kindNames.lookup({isMatmul, chain.bias != nullptr}).str() + actName.str();
  chain.kind = scratch;
  return chain;
}

// Clones `op`'s operand-producer subgraph (root [-> bias] -> act) into a
// fresh `fusion_group` region, remapping the chain's external inputs to
// block arguments (the region must be IsolatedFromAbove).
template <typename ActOp>
struct FuseActivation : public OpRewritePattern<ActOp> {
  FuseActivation(MLIRContext *ctx, StringRef actName)
      : OpRewritePattern<ActOp>(ctx), actName(actName) {}

  LogicalResult matchAndRewrite(ActOp actOp, PatternRewriter &rewriter) const override {
    auto chainOr = matchChain(actOp, actName);
    if (failed(chainOr)) return failure();
    MatchedChain chain = *chainOr;

    SmallVector<Value> externalInputs;
    if (auto mm = dyn_cast<MatmulOp>(chain.root)) {
      externalInputs = {mm.getLhs(), mm.getRhs()};
    } else {
      auto conv = cast<ConvOp>(chain.root);
      externalInputs = {conv.getInput(), conv.getFilter()};
    }
    Value biasExternal;
    if (chain.bias) {
      biasExternal = (chain.bias.getLhs() == chain.root->getResult(0))
                          ? chain.bias.getRhs() : chain.bias.getLhs();
      externalInputs.push_back(biasExternal);
    }

    auto group = rewriter.create<FusionGroupOp>(
        actOp.getLoc(), TypeRange{actOp.getResult().getType()},
        externalInputs, rewriter.getStringAttr(chain.kind));

    Block *body = rewriter.createBlock(&group.getBody());
    IRMapping mapping;
    for (Value ext : externalInputs)
      mapping.map(ext, body->addArgument(ext.getType(), ext.getLoc()));

    rewriter.setInsertionPointToStart(body);
    Operation *clonedRoot = rewriter.clone(*chain.root, mapping);
    mapping.map(chain.root->getResult(0), clonedRoot->getResult(0));

    Operation *lastCloned = clonedRoot;
    if (chain.bias) {
      Operation *clonedBias = rewriter.clone(*chain.bias, mapping);
      mapping.map(chain.bias.getResult(), clonedBias->getResult(0));
      lastCloned = clonedBias;
    }
    Operation *clonedAct = rewriter.clone(*actOp, mapping);
    rewriter.create<YieldOp>(actOp.getLoc(), ValueRange{clonedAct->getResult(0)});
    (void)lastCloned;

    rewriter.replaceOp(actOp, group.getResults());
    if (chain.bias) rewriter.eraseOp(chain.bias);
    rewriter.eraseOp(chain.root);
    return success();
  }

  StringRef actName;
};

struct FusionPass : public PassWrapper<FusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionPass)

  StringRef getArgument() const final { return "runtime-fusion"; }
  StringRef getDescription() const final {
    return "Fuse {matmul,conv}+bias+activation chains into runtime.fusion_group";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseActivation<ReluOp>>(&getContext(), "relu");
    patterns.add<FuseActivation<GeluOp>>(&getContext(), "gelu");
    patterns.add<FuseActivation<SigmoidOp>>(&getContext(), "sigmoid");
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFusionPass() { return std::make_unique<FusionPass>(); }

static PassRegistration<FusionPass> pass;

} // namespace runtime
