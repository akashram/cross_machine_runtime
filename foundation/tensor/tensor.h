#pragma once

// Unified Tensor Handle (v1) — CPU-only
// =========================================================================
//
// A TensorHandle is a lightweight, copyable descriptor for a multi-
// dimensional array.  It contains:
//
//   data_    — raw pointer to element [0, 0, ..., 0]
//   shape_   — size along each dimension
//   strides_ — byte offset between adjacent elements along each dim
//   dtype_   — element type (float32, float64, int32, ...)
//   device_  — where the data lives (CPU for now; GPU/FPGA in later phases)
//   buffer_  — shared_ptr owning the allocation (ref-count)
//
//
// Tensor handle vs. tensor value
// --------------------------------
// A TensorHandle is a handle to data, not a value that owns data uniquely.
// Copying a handle shares the underlying buffer (shared_ptr ref-count
// increments) without copying any elements.  This is intentional: view
// operations (transpose, slice, reshape) return new handles that share the
// same buffer — zero-copy.
//
//
// Shape and strides (layout)
// --------------------------
// Strides are in BYTES, not elements.  This matters when mixing dtypes or
// creating views over non-homogeneous memory.
//
// C-contiguous (row-major) layout: the last dimension changes fastest.
// For shape [M, N] with dtype float32 (4 bytes):
//   strides = [4*N, 4]  (stride along dim 0 = 4*N bytes; along dim 1 = 4 bytes)
//
// A transposed handle swaps strides without copying data.  A column-major
// (Fortran-order) tensor has strides in ascending order.
//
// is_contiguous() checks for C-order contiguity.  reshape() requires
// contiguity; transpose() and slice() produce non-contiguous views.
//
//
// Ownership model
// ---------------
// buffer_: std::shared_ptr<void> wraps the allocation.  Multiple handles
// can share the same buffer; it is freed when the last handle is destroyed.
// data_: a raw pointer within buffer_ (for slices, data_ != buffer_.get()).
//
// Non-owning view ("borrow"): buffer_ = nullptr, data_ = external pointer.
// The caller is responsible for the buffer lifetime.  use_count() returns 0.
// Use TensorHandle::borrow() for this pattern.
//
//
// No virtual dispatch
// -------------------
// TensorHandle uses a dtype enum and void* rather than a class hierarchy
// with virtual methods.  Type-safe access is exposed through templates at
// the call site:
//
//   float* p = t.data_as<float>();      // unchecked cast
//   float& v = t.at<float>({2, 3});     // bounds-checked element access
//
// Dispatch on dtype is done by the CALLER via switch(t.dtype()).  This
// matches how NumPy, PyTorch, and TF all work internally.
//
// Why not CRTP for device?  CRTP (TensorBase<DeviceTag>) would fix the
// device at compile time.  For a runtime system that moves tensors between
// devices dynamically, a runtime enum is more ergonomic.  Phase 3 will
// extend DeviceType with kCUDA; the handle layout is unchanged.
//
//
// Extension points for later phases
// -----------------------------------
// Phase 3 (GPU): extend DeviceType::kCUDA; alloc via cudaMalloc;
//   buffer_ destructor calls cudaFree.  The handle API is unchanged.
// Phase 5 (FPGA): extend DeviceType::kFPGA.
// Phase 6+: add quantisation metadata (scale, zero-point) as optional fields.
//
// =========================================================================

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

namespace foundation {

// =========================================================================
// Dtype
// =========================================================================
enum class Dtype : uint8_t {
    kFloat32,
    kFloat64,
    kFloat16,
    kBFloat16,
    kInt64,
    kInt32,
    kInt8,
    kUInt8,
};

inline std::size_t dtype_size(Dtype d) noexcept {
    switch (d) {
        case Dtype::kFloat64: return 8;
        case Dtype::kFloat32: return 4;
        case Dtype::kInt64:   return 8;
        case Dtype::kInt32:   return 4;
        case Dtype::kFloat16: return 2;
        case Dtype::kBFloat16:return 2;
        case Dtype::kInt8:    return 1;
        case Dtype::kUInt8:   return 1;
    }
    return 0;
}

inline const char* dtype_name(Dtype d) noexcept {
    switch (d) {
        case Dtype::kFloat32:  return "float32";
        case Dtype::kFloat64:  return "float64";
        case Dtype::kFloat16:  return "float16";
        case Dtype::kBFloat16: return "bfloat16";
        case Dtype::kInt64:    return "int64";
        case Dtype::kInt32:    return "int32";
        case Dtype::kInt8:     return "int8";
        case Dtype::kUInt8:    return "uint8";
    }
    return "unknown";
}

// =========================================================================
// Device
// =========================================================================
enum class DeviceType : uint8_t {
    kCPU  = 0,
    kCUDA = 1,  // stubbed; implemented in Phase 3
    kFPGA = 2,  // stubbed; implemented in Phase 5
};

inline const char* device_type_name(DeviceType t) noexcept {
    switch (t) {
        case DeviceType::kCPU:  return "cpu";
        case DeviceType::kCUDA: return "cuda";
        case DeviceType::kFPGA: return "fpga";
    }
    return "unknown";
}

struct Device {
    DeviceType type{DeviceType::kCPU};
    int        index{0};

    bool operator==(const Device&) const noexcept = default;
    bool operator!=(const Device&) const noexcept = default;
};

inline const Device kCPU{DeviceType::kCPU, 0};

// =========================================================================
// TensorHandle
// =========================================================================
class TensorHandle {
public:
    // Default: invalid / empty handle.
    TensorHandle() noexcept = default;

    // -----------------------------------------------------------------------
    // Factory functions
    // -----------------------------------------------------------------------

    // Allocate uninitialized storage; C-contiguous layout.
    static TensorHandle empty(std::span<const int64_t> shape,
                              Dtype dtype,
                              Device dev = kCPU) noexcept {
        int64_t n = numel_of(shape);
        std::size_t elem = dtype_size(dtype);
        std::size_t bytes = static_cast<std::size_t>(n) * elem;
        // malloc guarantees alignment to alignof(max_align_t) = 16 bytes on
        // most platforms, which covers all dtypes (max dtype_size = 8).
        // For larger alignments (cache lines, SIMD), use posix_memalign.
        void* raw = std::malloc(bytes == 0 ? 1 : bytes);
        if (!raw) return {};
        auto buf = std::shared_ptr<void>(raw, [](void* p){ std::free(p); });
        return TensorHandle(std::move(buf), raw,
                            to_vec(shape), contiguous_strides(shape, dtype),
                            dtype, dev);
    }

    // Allocate zero-initialised storage; C-contiguous layout.
    static TensorHandle zeros(std::span<const int64_t> shape,
                              Dtype dtype,
                              Device dev = kCPU) noexcept {
        auto t = empty(shape, dtype, dev);
        if (t.valid()) {
            std::size_t bytes = static_cast<std::size_t>(t.numel()) * dtype_size(dtype);
            std::memset(t.data_, 0, bytes);
        }
        return t;
    }

    // Wrap an existing allocation.  buffer keeps the allocation alive;
    // data points to element [0,...,0] (may be an offset into buffer).
    static TensorHandle from_buffer(std::shared_ptr<void> buffer,
                                    void* data,
                                    std::span<const int64_t> shape,
                                    std::span<const int64_t> strides,
                                    Dtype dtype,
                                    Device dev = kCPU) noexcept {
        return TensorHandle(std::move(buffer), data,
                            to_vec(shape), to_vec(strides),
                            dtype, dev);
    }

    // Non-owning view of external memory.  Caller must guarantee the
    // buffer outlives all handles created this way.  use_count() == 0.
    static TensorHandle borrow(void* data,
                               std::span<const int64_t> shape,
                               std::span<const int64_t> strides,
                               Dtype dtype,
                               Device dev = kCPU) noexcept {
        return TensorHandle(nullptr, data,
                            to_vec(shape), to_vec(strides),
                            dtype, dev);
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    bool    valid()         const noexcept { return data_ != nullptr; }
    int     ndim()          const noexcept { return static_cast<int>(shape_.size()); }
    Dtype   dtype()         const noexcept { return dtype_; }
    Device  device()        const noexcept { return device_; }
    void*   data()          const noexcept { return data_; }
    long    use_count()     const noexcept { return buffer_.use_count(); }

    int64_t size(int dim)   const noexcept {
        assert(dim >= 0 && dim < ndim());
        return shape_[static_cast<std::size_t>(dim)];
    }
    int64_t stride(int dim) const noexcept {
        assert(dim >= 0 && dim < ndim());
        return strides_[static_cast<std::size_t>(dim)];
    }
    int64_t numel() const noexcept { return numel_of(shape_); }

    const std::vector<int64_t>& shape()   const noexcept { return shape_; }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }

    // Unchecked typed pointer to the base element.
    template<typename T>
    T* data_as() const noexcept { return static_cast<T*>(data_); }

    // -----------------------------------------------------------------------
    // Layout queries
    // -----------------------------------------------------------------------

    // True iff the tensor is C-contiguous (row-major, no gaps or strides).
    bool is_contiguous() const noexcept {
        if (shape_.empty()) return true;
        int64_t expected = static_cast<int64_t>(dtype_size(dtype_));
        for (int i = ndim() - 1; i >= 0; --i) {
            if (shape_[static_cast<std::size_t>(i)] == 0) return true;
            if (strides_[static_cast<std::size_t>(i)] != expected) return false;
            expected *= shape_[static_cast<std::size_t>(i)];
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // Element access (bounds-checked in debug)
    // -----------------------------------------------------------------------
    // Indices are in logical order (dim 0 is the outermost).
    // The byte offset to element [i0, i1, ..., in] is sum(indices[k] * strides[k]).

    template<typename T>
    T& at(std::span<const int64_t> indices) noexcept {
        assert(sizeof(T) == dtype_size(dtype_));
        return *reinterpret_cast<T*>(byte_ptr(indices));
    }

    template<typename T>
    const T& at(std::span<const int64_t> indices) const noexcept {
        assert(sizeof(T) == dtype_size(dtype_));
        return *reinterpret_cast<const T*>(byte_ptr(indices));
    }

    // -----------------------------------------------------------------------
    // View operations — share the buffer, zero copy
    // -----------------------------------------------------------------------

    // Swap two dimensions (like numpy.transpose for 2-d, or any pair of dims).
    TensorHandle transpose(int d0, int d1) const noexcept {
        assert(valid() && d0 >= 0 && d0 < ndim() && d1 >= 0 && d1 < ndim());
        auto s  = shape_;
        auto st = strides_;
        std::swap(s [static_cast<std::size_t>(d0)], s [static_cast<std::size_t>(d1)]);
        std::swap(st[static_cast<std::size_t>(d0)], st[static_cast<std::size_t>(d1)]);
        return TensorHandle(buffer_, data_, std::move(s), std::move(st), dtype_, device_);
    }

    // Extract a contiguous range along one dimension.
    // Result has shape[dim] = ceil((end - start) / step).
    // data_ is offset to the first element; buffer_ keeps the allocation alive.
    TensorHandle slice(int dim, int64_t start, int64_t end,
                       int64_t step = 1) const noexcept {
        assert(valid());
        assert(dim >= 0 && dim < ndim());
        assert(step > 0 && start >= 0 && end >= start && end <= shape_[static_cast<std::size_t>(dim)]);
        auto s  = shape_;
        auto st = strides_;
        s [static_cast<std::size_t>(dim)] = (end - start + step - 1) / step;
        st[static_cast<std::size_t>(dim)] = strides_[static_cast<std::size_t>(dim)] * step;
        char* new_data = static_cast<char*>(data_) +
                         start * strides_[static_cast<std::size_t>(dim)];
        return TensorHandle(buffer_, new_data, std::move(s), std::move(st), dtype_, device_);
    }

    // Reinterpret the elements as a different shape.
    // Only valid for contiguous tensors; numel must match.
    TensorHandle reshape(std::span<const int64_t> new_shape) const noexcept {
        assert(valid() && is_contiguous());
        assert(numel_of(new_shape) == numel());
        return TensorHandle(buffer_, data_,
                            to_vec(new_shape),
                            contiguous_strides(new_shape, dtype_),
                            dtype_, device_);
    }

    // -----------------------------------------------------------------------
    // Debug
    // -----------------------------------------------------------------------
    void print_info() const noexcept {
        if (!valid()) { std::printf("TensorHandle(invalid)\n"); return; }
        std::printf("TensorHandle(dtype=%s device=%s:%d shape=[",
                    dtype_name(dtype_),
                    device_type_name(device_.type),
                    device_.index);
        for (int i = 0; i < ndim(); ++i)
            std::printf("%s%lld", i ? "," : "", static_cast<long long>(shape_[static_cast<std::size_t>(i)]));
        std::printf("] strides=[");
        for (int i = 0; i < ndim(); ++i)
            std::printf("%s%lld", i ? "," : "", static_cast<long long>(strides_[static_cast<std::size_t>(i)]));
        std::printf("] numel=%lld use_count=%ld contiguous=%s)\n",
                    static_cast<long long>(numel()),
                    use_count(),
                    is_contiguous() ? "yes" : "no");
    }

private:
    TensorHandle(std::shared_ptr<void> buf,
                 void* data,
                 std::vector<int64_t> shape,
                 std::vector<int64_t> strides,
                 Dtype dtype,
                 Device dev) noexcept
        : buffer_(std::move(buf))
        , data_(data)
        , shape_(std::move(shape))
        , strides_(std::move(strides))
        , dtype_(dtype)
        , device_(dev)
    {}

    // Compute C-contiguous strides (in bytes) from shape.
    static std::vector<int64_t> contiguous_strides(
            std::span<const int64_t> shape, Dtype dtype) noexcept {
        int n = static_cast<int>(shape.size());
        std::vector<int64_t> s(static_cast<std::size_t>(n));
        if (n == 0) return s;
        s[static_cast<std::size_t>(n - 1)] = static_cast<int64_t>(dtype_size(dtype));
        for (int i = n - 2; i >= 0; --i)
            s[static_cast<std::size_t>(i)] =
                s[static_cast<std::size_t>(i + 1)] * shape[static_cast<std::size_t>(i + 1)];
        return s;
    }

    static int64_t numel_of(std::span<const int64_t> shape) noexcept {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    static int64_t numel_of(const std::vector<int64_t>& shape) noexcept {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    static std::vector<int64_t> to_vec(std::span<const int64_t> s) noexcept {
        return {s.begin(), s.end()};
    }

    char* byte_ptr(std::span<const int64_t> indices) const noexcept {
        assert(static_cast<int>(indices.size()) == ndim());
        int64_t offset = 0;
        for (int i = 0; i < ndim(); ++i) {
            assert(indices[static_cast<std::size_t>(i)] >= 0 &&
                   indices[static_cast<std::size_t>(i)] < shape_[static_cast<std::size_t>(i)]);
            offset += indices[static_cast<std::size_t>(i)] * strides_[static_cast<std::size_t>(i)];
        }
        return static_cast<char*>(data_) + offset;
    }

    std::shared_ptr<void> buffer_;
    void*                 data_{nullptr};
    std::vector<int64_t>  shape_;
    std::vector<int64_t>  strides_;
    Dtype                 dtype_{Dtype::kFloat32};
    Device                device_{};
};

} // namespace foundation
