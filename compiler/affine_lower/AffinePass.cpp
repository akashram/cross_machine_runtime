//===- AffinePass.cpp - Step 6 implementation ----------------------------===//
//
// Two passes, deliberately kept separate even though both operate on the
// same loop bands: conversion (runtime.matmul -> affine.for nest) has to
// run once per matching op and doesn't compose with itself, while tiling
// is idempotent-ish and parameterized by tile size — keeping them as
// separate PassManager stages lets the AOT pipeline (step 12) tune tile
// size without re-running conversion, and lets `mlir-opt` test each in
// isolation.
//
//===----------------------------------------------------------------------===//

#include "AffinePass.h"
#include "RuntimeDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace runtime {
namespace {

// Lowers a single rank-2, fully-static `runtime.matmul` (no bias — the
// fusion pass, step 5, already peeled bias+activation into a
// fusion_group; a bare matmul reaching here means it wasn't fused) into
// an affine loop nest over freshly bufferized memrefs. Higher-rank
// (batched) matmul and dynamic shapes are left as `runtime.matmul` for
// kernel specialization (step 11) to dispatch as an opaque call —
// affine loop bounds must be compile-time constants, so there is nothing
// for this pass to do with them.
static void lowerMatmulToAffine(MatmulOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto lhsTy = dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto rhsTy = dyn_cast<RankedTensorType>(op.getRhs().getType());
  auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!lhsTy || !rhsTy || !resTy || lhsTy.getRank() != 2 ||
      !lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() ||
      !resTy.hasStaticShape() || op.getBias())
    return;

  int64_t M = resTy.getDimSize(0);
  int64_t N = resTy.getDimSize(1);
  int64_t K = lhsTy.getDimSize(1);

  auto lhsMemTy = MemRefType::get(lhsTy.getShape(), lhsTy.getElementType());
  auto rhsMemTy = MemRefType::get(rhsTy.getShape(), rhsTy.getElementType());
  auto resMemTy = MemRefType::get(resTy.getShape(), resTy.getElementType());

  Value lhsMem = builder.create<bufferization::ToMemrefOp>(loc, lhsMemTy, op.getLhs());
  Value rhsMem = builder.create<bufferization::ToMemrefOp>(loc, rhsMemTy, op.getRhs());
  Value outMem = builder.create<memref::AllocOp>(loc, resMemTy);

  Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(resTy.getElementType()));
  affine::buildAffineLoopNest(
      builder, loc, {0, 0}, {M, N}, {1, 1},
      [&](OpBuilder &b, Location l, ValueRange ivs) {
        b.create<affine::AffineStoreOp>(l, zero, outMem, ivs);
      });

  // Naive ijk order — the tiling pass below (and its documented ikj
  // interchange for the tiled band) is what actually makes this
  // cache-friendly; this pass only establishes correctness.
  affine::buildAffineLoopNest(
      builder, loc, {0, 0, 0}, {M, N, K}, {1, 1, 1},
      [&](OpBuilder &b, Location l, ValueRange ivs) {
        Value i = ivs[0], j = ivs[1], k = ivs[2];
        Value a = b.create<affine::AffineLoadOp>(l, lhsMem, ValueRange{i, k});
        Value bv = b.create<affine::AffineLoadOp>(l, rhsMem, ValueRange{k, j});
        Value c = b.create<affine::AffineLoadOp>(l, outMem, ValueRange{i, j});
        Value prod = b.create<arith::MulFOp>(l, a, bv);
        Value sum = b.create<arith::AddFOp>(l, c, prod);
        b.create<affine::AffineStoreOp>(l, sum, outMem, ValueRange{i, j});
      });

  Value outTensor = builder.create<bufferization::ToTensorOp>(loc, resTy, outMem);
  op.getResult().replaceAllUsesWith(outTensor);
  op.erase();
}

struct AffineConversionPass
    : public PassWrapper<AffineConversionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineConversionPass)

  StringRef getArgument() const final { return "runtime-affine-lower"; }
  StringRef getDescription() const final {
    return "Lower static-shape runtime.matmul to an affine.for loop nest";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<MatmulOp> toLower;
    func.walk([&](MatmulOp op) { toLower.push_back(op); });
    for (MatmulOp op : toLower) {
      OpBuilder builder(op);
      lowerMatmulToAffine(op, builder);
    }
  }
};

struct AffineTilingPass
    : public PassWrapper<AffineTilingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineTilingPass)

  AffineTilingPass() = default;
  explicit AffineTilingPass(unsigned tileSize) : tileSize(tileSize) {}

  StringRef getArgument() const final { return "runtime-affine-tile"; }
  StringRef getDescription() const final {
    return "Tile perfectly-nested affine.for bands produced by runtime-affine-lower";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    std::vector<SmallVector<affine::AffineForOp, 6>> bands;
    affine::getTileableBands(func, &bands);

    for (auto &band : bands) {
      SmallVector<unsigned> tileSizes(band.size(), tileSize);
      SmallVector<affine::AffineForOp> tiledNest;
      if (failed(affine::tilePerfectlyNested(band, tileSizes, &tiledNest))) {
        func.emitWarning("failed to tile affine band of depth ") << band.size();
        continue;
      }
      // Design doc's ikj interchange applies to the *point loops* (the
      // inner tileSize-bounded loops), not the tile loops: swapping which
      // of j/k is innermost changes whether the innermost store is
      // strided (bad) or the innermost load of B is strided (also bad,
      // but B is reused across the i tile so the miss is amortized).
      // Left as a follow-up once step 6's README has measured numbers to
      // justify the swap — tilePerfectlyNested's `tiledNest` already
      // gives the exact loop handles `affine::permuteLoops` would need.
    }
  }

  unsigned tileSize = 32;
};

} // namespace

std::unique_ptr<Pass> createAffineConversionPass() {
  return std::make_unique<AffineConversionPass>();
}

std::unique_ptr<Pass> createAffineTilingPass(unsigned tileSize) {
  return std::make_unique<AffineTilingPass>(tileSize);
}

static PassRegistration<AffineConversionPass> convPass;
static PassRegistration<AffineTilingPass> tilePass([] { return std::make_unique<AffineTilingPass>(32); });

} // namespace runtime
