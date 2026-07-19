//===- RuntimeDialect.cpp - `runtime` dialect registration --------------===//

#include "RuntimeDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RuntimeDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RuntimeAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "RuntimeOps.cpp.inc"

using namespace mlir;

namespace runtime {

void RuntimeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RuntimeOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "RuntimeAttrDefs.cpp.inc"
      >();
}

LogicalResult ShardSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                     int64_t numDevices,
                                     llvm::ArrayRef<int64_t> partitions) {
  if (numDevices <= 0)
    return emitError() << "shard_spec numDevices must be positive, got " << numDevices;
  int64_t product = 1;
  for (int64_t p : partitions) {
    if (p == -1) continue; // replicated along this dim
    if (p <= 0)
      return emitError() << "shard_spec partition factor must be -1 (replicated) or positive";
    product *= p;
  }
  if (product > numDevices)
    return emitError() << "shard_spec partitions multiply to " << product
                        << ", exceeding numDevices=" << numDevices;
  return success();
}

//===----------------------------------------------------------------------===//
// InferTypeOpInterface implementations
//
// Declared via DeclareOpInterfaceMethods<InferTypeOpInterface> in
// RuntimeOps.td (which only emits the method *declaration*); definitions
// live here because ODS has no notion of "output shape = f(input shapes)"
// beyond the trivial same-operand-and-result-type case (handled separately
// via the SameOperandsAndResultType trait for Relu/Gelu/Sigmoid/Softmax).
//===----------------------------------------------------------------------===//

// Shared helper: broadcast two ranked shapes numpy-style, or bail to a
// fully-dynamic result if either operand is unranked / rank differs in a
// way that can't be resolved statically. The shape inference pass (step 4)
// re-invokes this once more input shapes are known, so a conservative
// dynamic result here is refined, not final.
static Type inferElementwiseResult(Value lhs, Value rhs) {
  auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsTy || !rhsTy)
    return lhs.getType();
  if (lhsTy.getRank() != rhsTy.getRank())
    return RankedTensorType::get(
        SmallVector<int64_t>(std::max(lhsTy.getRank(), rhsTy.getRank()),
                              ShapedType::kDynamic),
        lhsTy.getElementType());
  SmallVector<int64_t> outShape;
  outShape.reserve(lhsTy.getRank());
  for (auto [l, r] : llvm::zip(lhsTy.getShape(), rhsTy.getShape())) {
    if (l == r) outShape.push_back(l);
    else if (l == 1) outShape.push_back(r);
    else if (r == 1) outShape.push_back(l);
    else outShape.push_back(ShapedType::kDynamic); // conflicting static dims
  }
  return RankedTensorType::get(outShape, lhsTy.getElementType());
}

LogicalResult MatmulOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type> &inferred) {
  Adaptor adaptor(operands, attrs, {}, {});
  auto lhsTy = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhsTy = dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
  if (!lhsTy || !rhsTy || lhsTy.getRank() < 2 || rhsTy.getRank() < 2) {
    inferred.push_back(adaptor.getLhs().getType());
    return success();
  }
  // Batch dims are everything before the trailing 2; M/K/N come from the
  // last two dims, swapped if the corresponding transpose_* flag is set.
  SmallVector<int64_t> outShape(lhsTy.getShape().drop_back(2));
  int64_t m = adaptor.getTransposeLhs() ? lhsTy.getDimSize(lhsTy.getRank() - 1)
                                         : lhsTy.getDimSize(lhsTy.getRank() - 2);
  int64_t n = adaptor.getTransposeRhs() ? rhsTy.getDimSize(rhsTy.getRank() - 2)
                                         : rhsTy.getDimSize(rhsTy.getRank() - 1);
  outShape.push_back(m);
  outShape.push_back(n);
  inferred.push_back(RankedTensorType::get(outShape, lhsTy.getElementType()));
  return success();
}

LogicalResult ConvOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type> &inferred) {
  Adaptor adaptor(operands, attrs, {}, {});
  auto inTy = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto filterTy = dyn_cast<RankedTensorType>(adaptor.getFilter().getType());
  if (!inTy || !filterTy || inTy.getRank() != 4 || filterTy.getRank() != 4) {
    inferred.push_back(adaptor.getInput().getType());
    return success();
  }
  // NCHW input, [outC, inC, kH, kW] filter.
  auto strides = llvm::to_vector(adaptor.getStrides().getAsValueRange<IntegerAttr>());
  auto padding = llvm::to_vector(adaptor.getPadding().getAsValueRange<IntegerAttr>());
  auto dilations = llvm::to_vector(adaptor.getDilations().getAsValueRange<IntegerAttr>());
  auto spatialOut = [&](int64_t inDim, int64_t kDim, int idx) -> int64_t {
    if (ShapedType::isDynamic(inDim)) return ShapedType::kDynamic;
    int64_t effK = (kDim - 1) * dilations[idx].getSExtValue() + 1;
    return (inDim + 2 * padding[idx].getSExtValue() - effK) / strides[idx].getSExtValue() + 1;
  };
  SmallVector<int64_t> outShape = {
      inTy.getDimSize(0), filterTy.getDimSize(0),
      spatialOut(inTy.getDimSize(2), filterTy.getDimSize(2), 0),
      spatialOut(inTy.getDimSize(3), filterTy.getDimSize(3), 1)};
  inferred.push_back(RankedTensorType::get(outShape, inTy.getElementType()));
  return success();
}

#define ELEMENTWISE_UNARY_INFER(OpName)                                      \
  LogicalResult OpName::inferReturnTypes(                                    \
      MLIRContext *, std::optional<Location>, ValueRange operands,           \
      DictionaryAttr, OpaqueProperties, RegionRange,                         \
      SmallVectorImpl<Type> &inferred) {                                     \
    inferred.push_back(operands.front().getType());                         \
    return success();                                                        \
  }
ELEMENTWISE_UNARY_INFER(ReluOp)
ELEMENTWISE_UNARY_INFER(GeluOp)
ELEMENTWISE_UNARY_INFER(SigmoidOp)
#undef ELEMENTWISE_UNARY_INFER

#define ELEMENTWISE_BINARY_INFER(OpName)                                     \
  LogicalResult OpName::inferReturnTypes(                                    \
      MLIRContext *, std::optional<Location>, ValueRange operands,           \
      DictionaryAttr, OpaqueProperties, RegionRange,                         \
      SmallVectorImpl<Type> &inferred) {                                     \
    inferred.push_back(inferElementwiseResult(operands[0], operands[1]));   \
    return success();                                                        \
  }
ELEMENTWISE_BINARY_INFER(AddOp)
ELEMENTWISE_BINARY_INFER(MulOp)
ELEMENTWISE_BINARY_INFER(SubOp)
#undef ELEMENTWISE_BINARY_INFER

DeviceKind getAssignedDevice(Operation *op) {
  if (auto attr = op->getAttrOfType<DeviceAttr>(kDeviceAttrName))
    return attr.getValue();
  return DeviceKind::Unassigned;
}

void setAssignedDevice(Operation *op, DeviceKind device) {
  op->setAttr(kDeviceAttrName,
              DeviceAttr::get(op->getContext(), device));
}

bool isFusable(Operation *op) {
  // Elementwise + matmul + reduce are fusable; ops with region/side effects
  // (fusion_group itself, kernel_call, transfer, collectives) are not —
  // they're either already the output of fusion or crosses a device/shard
  // boundary that fusion must not paper over.
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<MatmulOp, ConvOp, AddOp, MulOp, SubOp, ReluOp, GeluOp, SigmoidOp,
            SoftmaxOp, ReduceOp>([](Operation *) { return true; })
      .Default([](Operation *) { return false; });
}

} // namespace runtime
