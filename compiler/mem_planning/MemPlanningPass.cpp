//===- MemPlanningPass.cpp - Step 7 implementation -----------------------===//
//
// Classic interval-based buffer assignment (the same shape as XLA's
// BufferAssignment / TF's memory optimizer): treat program order as a
// timeline, compute each value's [def, last-use] interval, and greedily
// place intervals into a shared arena with a first-fit free list so that
// overlapping intervals never alias and disjoint ones reuse bytes. This
// pass answers "how much peak memory" and "which offset" — it does *not*
// decide *whether* to keep a value alive longer (that's rematerialization,
// step 8, which runs after this pass and can shrink the intervals this
// pass sees on a second run).
//
//===----------------------------------------------------------------------===//

#include "MemPlanningPass.h"
#include "RuntimeDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>

using namespace mlir;

namespace runtime {
namespace {

constexpr llvm::StringLiteral kOffsetAttr = "runtime.buffer_offset";
constexpr llvm::StringLiteral kPeakMemAttr = "runtime.peak_memory_bytes";

static int64_t byteSize(RankedTensorType ty) {
  int64_t elems = 1;
  for (int64_t d : ty.getShape()) elems *= d; // caller guarantees static
  unsigned bits = ty.getElementType().getIntOrFloatBitWidth();
  return elems * static_cast<int64_t>((bits + 7) / 8);
}

struct Interval {
  Value value;
  Operation *definingOp;
  int64_t defTime;
  int64_t lastUseTime;
  int64_t size;
};

// One block of the shared arena, either live (offset owned by `owner`) or
// on the free list.
struct FreeBlock {
  int64_t offset;
  int64_t size;
};

struct MemPlanningPass
    : public PassWrapper<MemPlanningPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemPlanningPass)

  StringRef getArgument() const final { return "runtime-mem-planning"; }
  StringRef getDescription() const final {
    return "Assign shared-arena buffer offsets via interval liveness analysis";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Block &body = func.getBody().front();

    // Program-order timestamp per op, and per-value liveness intervals for
    // every statically-shaped tensor result.
    llvm::DenseMap<Operation *, int64_t> timeOf;
    SmallVector<Interval> intervals;
    int64_t t = 0;
    for (Operation &op : body) {
      timeOf[&op] = t++;
      for (Value result : op.getResults()) {
        auto ty = dyn_cast<RankedTensorType>(result.getType());
        if (!ty || !ty.hasStaticShape()) continue; // remat/dynamic: not planned
        intervals.push_back({result, &op, timeOf[&op], timeOf[&op], byteSize(ty)});
      }
    }
    for (Interval &iv : intervals) {
      int64_t lastUse = iv.defTime;
      for (Operation *user : iv.value.getUsers())
        lastUse = std::max(lastUse, timeOf.lookup(user));
      iv.lastUseTime = lastUse;
    }

    // Sweep in def-time order; two events per interval (alloc at defTime,
    // free at lastUseTime) processed via re-sorting the still-live set
    // each step is O(n^2 log n) but n is "tensor values in one function",
    // in the hundreds at most — clarity over asymptotics here.
    llvm::stable_sort(intervals, [](const Interval &a, const Interval &b) {
      return a.defTime < b.defTime;
    });

    SmallVector<FreeBlock> freeList = {{0, INT64_MAX}};
    llvm::DenseMap<Value, std::pair<int64_t, int64_t>> placement; // value -> (offset, size)
    SmallVector<Interval *> live;
    int64_t peak = 0;

    auto releaseExpired = [&](int64_t now) {
      llvm::erase_if(live, [&](Interval *iv) {
        if (iv->lastUseTime >= now) return false;
        auto [offset, size] = placement[iv->value];
        freeList.push_back({offset, size});
        return true;
      });
      llvm::sort(freeList, [](const FreeBlock &a, const FreeBlock &b) {
        return a.offset < b.offset;
      });
      // Coalesce adjacent free blocks so long-lived small values don't
      // fragment the arena into unusable slivers.
      for (size_t i = 0; i + 1 < freeList.size();) {
        if (freeList[i].offset + freeList[i].size == freeList[i + 1].offset) {
          freeList[i].size += freeList[i + 1].size;
          freeList.erase(freeList.begin() + i + 1);
        } else {
          ++i;
        }
      }
    };

    for (Interval &iv : intervals) {
      releaseExpired(iv.defTime);

      // First-fit: smallest-offset free block that's big enough.
      auto it = llvm::find_if(freeList, [&](const FreeBlock &b) { return b.size >= iv.size; });
      int64_t offset = it->offset;
      it->offset += iv.size;
      it->size -= iv.size;
      if (it->size == 0) freeList.erase(it);

      placement[iv.value] = {offset, iv.size};
      live.push_back(&iv);
      peak = std::max(peak, offset + iv.size);

      iv.definingOp->setAttr(kOffsetAttr, Builder(&getContext()).getI64IntegerAttr(offset));
    }

    func->setAttr(kPeakMemAttr, Builder(&getContext()).getI64IntegerAttr(peak));
  }
};

} // namespace

std::unique_ptr<Pass> createMemoryPlanningPass() {
  return std::make_unique<MemPlanningPass>();
}

static PassRegistration<MemPlanningPass> pass;

} // namespace runtime
