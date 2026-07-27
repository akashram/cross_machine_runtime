//===- StableHLOLowerPass.cpp - Step 3 implementation --------------------===//
//
// Unrun here: needs an MLIR build with the StableHLO dialect registered
// (github.com/openxla/stablehlo, built against the same LLVM commit as
// compiler/mlir_setup's build) — no MLIR/LLVM toolchain on this Mac, same
// gate as the rest of compiler/. Written in full, not stubbed, matching
// Phase 3/4's convention for hardware/toolchain-gated code.
//
//===----------------------------------------------------------------------===//

#include "StableHLOLowerPass.h"
#include "RuntimeDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace runtime {
namespace {

//===----------------------------------------------------------------------===//
// runtime.matmul -> stablehlo.dot_general
//
// `transpose_lhs`/`transpose_rhs` become the contracting-dimension index
// (last dim if not transposed, second-to-last if transposed) rather than an
// actual transpose op — dot_general takes arbitrary contracting dims, so
// this avoids materializing a transposed copy XLA would just optimize away.
//===----------------------------------------------------------------------===//

struct LowerMatmul : public OpConversionPattern<MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getBias())
      return rewriter.notifyMatchFailure(op, "fused bias not supported — "
          "run FusionPass's inverse (dissolve fusion_group) first so bias "
          "shows up as a separate runtime.add for this pass to lower");

    auto lhsType = cast<RankedTensorType>(adaptor.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(adaptor.getRhs().getType());
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();

    int64_t lhsContract = adaptor.getTransposeLhs() ? lhsRank - 2 : lhsRank - 1;
    int64_t rhsContract = adaptor.getTransposeRhs() ? rhsRank - 1 : rhsRank - 2;

    // Every leading dim before the matmul's own two is a batch dim,
    // matching runtime.matmul's "batched matrix multiply" semantics.
    SmallVector<int64_t> batchDims;
    for (int64_t i = 0; i + 2 < lhsRank; ++i) batchDims.push_back(i);

    auto dotDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*lhsBatchingDimensions=*/batchDims,
        /*rhsBatchingDimensions=*/batchDims,
        /*lhsContractingDimensions=*/{lhsContract},
        /*rhsContractingDimensions=*/{rhsContract});

    rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        op, op.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        dotDims, /*precisionConfig=*/nullptr, /*algorithm=*/nullptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Elementwise binary/unary ops — near-direct renames.
//===----------------------------------------------------------------------===//

template <typename RuntimeOp, typename StableHLOOp>
struct LowerBinary : public OpConversionPattern<RuntimeOp> {
  using OpConversionPattern<RuntimeOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<RuntimeOp>::OpAdaptor;

  LogicalResult matchAndRewrite(RuntimeOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<StableHLOOp>(op, op.getResult().getType(),
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// stablehlo has no direct `relu`; lower to max(x, 0) against a splat zero,
// the standard StableHLO idiom (matches what jax.nn.relu itself lowers to).
struct LowerRelu : public OpConversionPattern<ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto zero = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), SplatElementsAttr::get(resultType, rewriter.getZeroAttr(resultType.getElementType())));
    rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(op, resultType, adaptor.getInput(), zero);
    return success();
  }
};

struct LowerSigmoid : public OpConversionPattern<SigmoidOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SigmoidOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    // stablehlo.logistic is exactly sigmoid; a direct rename.
    rewriter.replaceOpWithNewOp<stablehlo::LogisticOp>(op, op.getResult().getType(), adaptor.getInput());
    return success();
  }
};

// GELU has no single StableHLO op. Lower the tanh approximation
// (0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))) used by most JAX/Flax models,
// documented here rather than left implicit since it's a numerical choice
// (the exact erf-based GELU would need stablehlo.custom_call to an erf impl).
struct LowerGelu : public OpConversionPattern<GeluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GeluOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elemType = resultType.getElementType();
    Value x = adaptor.getInput();

    auto splatConst = [&](double v) {
      return rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(resultType, rewriter.getFloatAttr(elemType, v)));
    };
    Value c0_044715 = splatConst(0.044715);
    Value cSqrt2OverPi = splatConst(0.7978845608028654);
    Value cHalf = splatConst(0.5);
    Value cOne = splatConst(1.0);

    Value x3 = rewriter.create<stablehlo::MulOp>(loc, resultType, x,
                   rewriter.create<stablehlo::MulOp>(loc, resultType, x, x));
    Value inner = rewriter.create<stablehlo::AddOp>(loc, resultType, x,
                      rewriter.create<stablehlo::MulOp>(loc, resultType, c0_044715, x3));
    Value scaled = rewriter.create<stablehlo::MulOp>(loc, resultType, cSqrt2OverPi, inner);
    Value tanh = rewriter.create<stablehlo::TanhOp>(loc, resultType, scaled);
    Value onePlusTanh = rewriter.create<stablehlo::AddOp>(loc, resultType, cOne, tanh);
    Value halfX = rewriter.create<stablehlo::MulOp>(loc, resultType, cHalf, x);
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, resultType, halfX, onePlusTanh);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// runtime.softmax -> exp(x - max(x)) / sum(exp(x - max(x))), the standard
// numerically-stable decomposition. reduce ops need an explicit combiner
// region, factored into buildReduce() and reused by runtime.reduce below.
//===----------------------------------------------------------------------===//

static stablehlo::ReduceOp buildReduce(ConversionPatternRewriter &rewriter, Location loc,
                                        Value input, ArrayRef<int64_t> axes,
                                        RankedTensorType resultType, bool isMax) {
  Type elemType = resultType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);
  Attribute initAttr = isMax
      ? (Attribute)DenseElementsAttr::get(scalarType, rewriter.getFloatAttr(elemType, -std::numeric_limits<double>::infinity()))
      : (Attribute)DenseElementsAttr::get(scalarType, rewriter.getZeroAttr(elemType));
  Value init = rewriter.create<stablehlo::ConstantOp>(loc, initAttr);

  auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
      loc, TypeRange{resultType}, ValueRange{input}, ValueRange{init},
      rewriter.getI64TensorAttr(axes));
  Block *body = rewriter.createBlock(&reduceOp.getBody());
  body->addArgument(scalarType, loc);
  body->addArgument(scalarType, loc);
  rewriter.setInsertionPointToStart(body);
  Value combined = isMax
      ? (Value)rewriter.create<stablehlo::MaxOp>(loc, scalarType, body->getArgument(0), body->getArgument(1))
      : (Value)rewriter.create<stablehlo::AddOp>(loc, scalarType, body->getArgument(0), body->getArgument(1));
  rewriter.create<stablehlo::ReturnOp>(loc, ValueRange{combined});
  rewriter.setInsertionPointAfter(reduceOp);
  return reduceOp;
}

struct LowerSoftmax : public OpConversionPattern<SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    int64_t axis = adaptor.getAxis();

    SmallVector<int64_t> reducedShape(inputType.getShape());
    reducedShape.erase(reducedShape.begin() + axis);
    auto reducedType = RankedTensorType::get(reducedShape, inputType.getElementType());

    Value maxVal = buildReduce(rewriter, loc, adaptor.getInput(), {axis}, reducedType, /*isMax=*/true);
    Value maxBcast = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, inputType, maxVal, broadcastDimsSkipping(rewriter, inputType.getRank(), axis));
    Value shifted = rewriter.create<stablehlo::SubtractOp>(loc, inputType, adaptor.getInput(), maxBcast);
    Value expVal = rewriter.create<stablehlo::ExpOp>(loc, inputType, shifted);

    Value sumVal = buildReduce(rewriter, loc, expVal, {axis}, reducedType, /*isMax=*/false);
    Value sumBcast = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, inputType, sumVal, broadcastDimsSkipping(rewriter, inputType.getRank(), axis));
    rewriter.replaceOpWithNewOp<stablehlo::DivOp>(op, inputType, expVal, sumBcast);
    return success();
  }

  static DenseIntElementsAttr broadcastDimsSkipping(PatternRewriter &rewriter, int64_t rank, int64_t skipAxis) {
    SmallVector<int64_t> dims;
    for (int64_t i = 0; i < rank; ++i)
      if (i != skipAxis) dims.push_back(i);
    return rewriter.getI64TensorAttr(dims);
  }
};

//===----------------------------------------------------------------------===//
// runtime.reduce (sum/max/mean) -> stablehlo.reduce (+ a scalar divide for
// mean, since StableHLO has no reduce-mean primitive).
//===----------------------------------------------------------------------===//

struct LowerReduce : public OpConversionPattern<ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> axes;
    for (Attribute a : adaptor.getAxes())
      axes.push_back(cast<IntegerAttr>(a).getInt());

    // keep_dims is handled by a trailing reshape — stablehlo.reduce always
    // drops the reduced dims, same as runtime.reduce with keep_dims=false.
    SmallVector<int64_t> reducedShape(inputType.getShape());
    for (int64_t ax : llvm::reverse(axes)) reducedShape.erase(reducedShape.begin() + ax);
    auto reducedType = RankedTensorType::get(reducedShape, inputType.getElementType());

    ReduceKind kind = adaptor.getKind();
    bool isMax = kind == ReduceKind::MAX;
    Value reduced = buildReduce(rewriter, loc, adaptor.getInput(), axes, reducedType, isMax);

    if (kind == ReduceKind::MEAN) {
      int64_t count = 1;
      for (int64_t ax : axes) count *= inputType.getDimSize(ax);
      Value countConst = rewriter.create<stablehlo::ConstantOp>(
          loc, SplatElementsAttr::get(reducedType, rewriter.getFloatAttr(reducedType.getElementType(), (double)count)));
      reduced = rewriter.create<stablehlo::DivOp>(loc, reducedType, reduced, countConst);
    }

    if (adaptor.getKeepDims())
      reduced = rewriter.create<stablehlo::ReshapeOp>(loc, resultType, reduced);
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// runtime.conv -> stablehlo.convolution, NCHW/NCHW/NCHW dimension numbers
// (matches runtime.conv's documented NCHW layout, so no transpose needed).
//===----------------------------------------------------------------------===//

struct LowerConv : public OpConversionPattern<ConvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ConvOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getBias())
      return rewriter.notifyMatchFailure(op, "fused bias not supported, same as LowerMatmul");

    auto dimNums = stablehlo::ConvDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*inputBatchDimension=*/0, /*inputFeatureDimension=*/1,
        /*inputSpatialDimensions=*/{2, 3},
        /*kernelInputFeatureDimension=*/1, /*kernelOutputFeatureDimension=*/0,
        /*kernelSpatialDimensions=*/{2, 3},
        /*outputBatchDimension=*/0, /*outputFeatureDimension=*/1,
        /*outputSpatialDimensions=*/{2, 3});

    SmallVector<int64_t> strides, padLo, padHi, dilations;
    for (Attribute a : adaptor.getStrides()) strides.push_back(cast<IntegerAttr>(a).getInt());
    for (Attribute a : adaptor.getPadding()) { padLo.push_back(cast<IntegerAttr>(a).getInt()); padHi.push_back(cast<IntegerAttr>(a).getInt()); }
    for (Attribute a : adaptor.getDilations()) dilations.push_back(cast<IntegerAttr>(a).getInt());

    SmallVector<int64_t> paddingPairs;
    for (size_t i = 0; i < padLo.size(); ++i) { paddingPairs.push_back(padLo[i]); paddingPairs.push_back(padHi[i]); }
    auto paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({(int64_t)padLo.size(), 2}, rewriter.getI64Type()), paddingPairs);

    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        op, op.getResult().getType(), adaptor.getInput(), adaptor.getFilter(),
        rewriter.getI64TensorAttr(strides), paddingAttr,
        /*lhsDilation=*/nullptr, rewriter.getI64TensorAttr(dilations),
        /*windowReversal=*/nullptr, dimNums,
        /*featureGroupCount=*/rewriter.getI64IntegerAttr(1),
        /*batchGroupCount=*/rewriter.getI64IntegerAttr(1),
        /*precisionConfig=*/nullptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// runtime.all_gather / runtime.reduce_scatter -> stablehlo's ICI-native
// collectives. This is what step 8 (ici_collectives) executes: the runtime
// dialect's collective ops, chosen deliberately to name-match XLA's own
// collective ops, lower here nearly 1:1.
//===----------------------------------------------------------------------===//

struct LowerAllGather : public OpConversionPattern<AllGatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(AllGatherOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t> replicaGroups;
    for (int64_t i = 0; i < adaptor.getNumShards(); ++i) replicaGroups.push_back(i);
    auto groupsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1, adaptor.getNumShards()}, rewriter.getI64Type()), replicaGroups);

    rewriter.replaceOpWithNewOp<stablehlo::AllGatherOp>(
        op, op.getResult().getType(), adaptor.getInput(),
        rewriter.getI64IntegerAttr(adaptor.getShardDim()), groupsAttr,
        /*channelHandle=*/nullptr, /*useGlobalDeviceIds=*/nullptr);
    return success();
  }
};

// runtime.reduce_scatter has no reduction-kind attribute (sum is the only
// use PLAN.md's steps need — gradient/activation all-reduce), so the
// combiner region below is hardcoded to add, unlike LowerReduce above.
struct LowerReduceScatter : public OpConversionPattern<ReduceScatterOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReduceScatterOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elemType = resultType.getElementType();

    SmallVector<int64_t> replicaGroups;
    for (int64_t i = 0; i < adaptor.getNumShards(); ++i) replicaGroups.push_back(i);
    auto groupsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({1, adaptor.getNumShards()}, rewriter.getI64Type()), replicaGroups);

    auto rsOp = rewriter.create<stablehlo::ReduceScatterOp>(
        loc, resultType, adaptor.getInput(), rewriter.getI64IntegerAttr(adaptor.getShardDim()),
        groupsAttr, /*channelHandle=*/nullptr, /*useGlobalDeviceIds=*/nullptr);
    Block *body = rewriter.createBlock(&rsOp.getComputation());
    auto scalarType = RankedTensorType::get({}, elemType);
    body->addArgument(scalarType, loc);
    body->addArgument(scalarType, loc);
    rewriter.setInsertionPointToStart(body);
    Value sum = rewriter.create<stablehlo::AddOp>(loc, scalarType, body->getArgument(0), body->getArgument(1));
    rewriter.create<stablehlo::ReturnOp>(loc, ValueRange{sum});

    rewriter.replaceOp(op, rsOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct StableHLOLowerPass : public PassWrapper<StableHLOLowerPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StableHLOLowerPass)

  StringRef getArgument() const final { return "runtime-to-stablehlo"; }
  StringRef getDescription() const final {
    return "Lower the runtime dialect to StableHLO for TPU (JAX/XLA) execution";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<stablehlo::StablehloDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addIllegalDialect<RuntimeDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatmul>(&getContext());
    patterns.add<LowerBinary<AddOp, stablehlo::AddOp>>(&getContext());
    patterns.add<LowerBinary<MulOp, stablehlo::MulOp>>(&getContext());
    patterns.add<LowerBinary<SubOp, stablehlo::SubtractOp>>(&getContext());
    patterns.add<LowerRelu>(&getContext());
    patterns.add<LowerSigmoid>(&getContext());
    patterns.add<LowerGelu>(&getContext());
    patterns.add<LowerSoftmax>(&getContext());
    patterns.add<LowerReduce>(&getContext());
    patterns.add<LowerConv>(&getContext());
    patterns.add<LowerAllGather>(&getContext());
    patterns.add<LowerReduceScatter>(&getContext());
    // runtime.gather/scatter, runtime.transfer, runtime.kernel_call: no
    // pattern yet. Gather/scatter need stablehlo.gather/scatter's full
    // dimension-numbers + (for scatter) a combiner region — real work, left
    // TODO rather than a shallow approximation that would silently produce
    // wrong indices. transfer/kernel_call are placement/kernel-specialization
    // artifacts that shouldn't exist on the TPU path (XLA does its own
    // placement within a slice); reaching this pass means an earlier pass
    // ran in the wrong order, so they're deliberately left illegal to catch
    // that as a conversion failure instead of silently mis-lowering.

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createStableHLOLowerPass() { return std::make_unique<StableHLOLowerPass>(); }

static PassRegistration<StableHLOLowerPass> pass;

} // namespace runtime
