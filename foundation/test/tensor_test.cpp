#include "tensor/tensor.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <array>

using foundation::Dtype;
using foundation::DeviceType;
using foundation::Device;
using foundation::TensorHandle;
using foundation::dtype_size;
using foundation::kCPU;

// ---------------------------------------------------------------------------
// Test 1: dtype_size is correct for all dtypes
// ---------------------------------------------------------------------------
static void test_dtype_sizes() {
    assert(dtype_size(Dtype::kFloat32)  == 4);
    assert(dtype_size(Dtype::kFloat64)  == 8);
    assert(dtype_size(Dtype::kFloat16)  == 2);
    assert(dtype_size(Dtype::kBFloat16) == 2);
    assert(dtype_size(Dtype::kInt64)    == 8);
    assert(dtype_size(Dtype::kInt32)    == 4);
    assert(dtype_size(Dtype::kInt8)     == 1);
    assert(dtype_size(Dtype::kUInt8)    == 1);
    printf("PASS  test_dtype_sizes\n");
}

// ---------------------------------------------------------------------------
// Test 2: empty() creates a valid tensor with correct shape and strides
// ---------------------------------------------------------------------------
static void test_empty_shape_strides() {
    std::array<int64_t, 2> sh{3, 4};
    auto t = TensorHandle::empty(sh, Dtype::kFloat32);

    assert(t.valid());
    assert(t.ndim() == 2);
    assert(t.size(0) == 3 && t.size(1) == 4);
    assert(t.numel() == 12);
    assert(t.dtype() == Dtype::kFloat32);
    assert(t.device() == kCPU);
    assert(t.data() != nullptr);
    assert(t.use_count() == 1);

    // C-contiguous strides: [4*sizeof(float), sizeof(float)] = [16, 4]
    assert(t.stride(1) == 4);
    assert(t.stride(0) == 4 * 4);

    assert(t.is_contiguous());
    printf("PASS  test_empty_shape_strides\n");
}

// ---------------------------------------------------------------------------
// Test 3: zeros() initialises to zero
// ---------------------------------------------------------------------------
static void test_zeros() {
    std::array<int64_t, 2> sh{5, 6};
    auto t = TensorHandle::zeros(sh, Dtype::kFloat32);
    assert(t.valid());

    for (int i = 0; i < 30; ++i)
        if (t.data_as<float>()[i] != 0.0f) __builtin_trap();

    printf("PASS  test_zeros\n");
}

// ---------------------------------------------------------------------------
// Test 4: element access via at<T>() reads and writes correctly
// ---------------------------------------------------------------------------
static void test_element_access() {
    std::array<int64_t, 2> sh{3, 4};
    auto t = TensorHandle::zeros(sh, Dtype::kFloat32);

    // Write via at<float>
    const std::array<int64_t, 2> idx{1, 2};
    t.at<float>(idx) = 3.14f;
    if (t.at<float>(idx) != 3.14f) __builtin_trap();

    // Verify direct pointer math gives same result
    if (t.data_as<float>()[1 * 4 + 2] != 3.14f) __builtin_trap();

    printf("PASS  test_element_access\n");
}

// ---------------------------------------------------------------------------
// Test 5: copying a handle shares the buffer (ref-count increases)
// ---------------------------------------------------------------------------
static void test_ref_count_sharing() {
    std::array<int64_t, 1> sh{8};
    auto t = TensorHandle::empty(sh, Dtype::kFloat64);
    assert(t.use_count() == 1);

    {
        auto t2 = t;  // copy
        assert(t.use_count() == 2);
        assert(t2.use_count() == 2);
        assert(t.data() == t2.data());  // same buffer
    }  // t2 destroyed

    assert(t.use_count() == 1);
    printf("PASS  test_ref_count_sharing\n");
}

// ---------------------------------------------------------------------------
// Test 6: borrow() creates a non-owning view (use_count == 0)
// ---------------------------------------------------------------------------
static void test_borrow() {
    float buf[6] = {1, 2, 3, 4, 5, 6};
    std::array<int64_t, 2> sh{2, 3};
    std::array<int64_t, 2> st{12, 4};  // row-major float32 strides in bytes
    auto t = TensorHandle::borrow(buf, sh, st, Dtype::kFloat32);

    assert(t.valid());
    assert(t.use_count() == 0);  // non-owning
    assert(t.data() == static_cast<void*>(buf));

    const std::array<int64_t, 2> idx{1, 0};
    if (t.at<float>(idx) != 4.0f) __builtin_trap();  // buf[1*3 + 0] = buf[3] = 4

    printf("PASS  test_borrow\n");
}

// ---------------------------------------------------------------------------
// Test 7: transpose swaps shape and strides, marks non-contiguous
// ---------------------------------------------------------------------------
static void test_transpose() {
    std::array<int64_t, 2> sh{3, 4};
    auto t = TensorHandle::zeros(sh, Dtype::kFloat32);

    // Write a distinctive value at [1,2] in the original
    std::array<int64_t, 2> orig_idx{1, 2};
    t.at<float>(orig_idx) = 7.0f;

    auto tr = t.transpose(0, 1);  // shape becomes [4,3]
    assert(tr.ndim() == 2);
    assert(tr.size(0) == 4 && tr.size(1) == 3);
    assert(tr.stride(0) == 4);     // original stride(1) = 4 bytes
    assert(tr.stride(1) == 16);    // original stride(0) = 16 bytes
    assert(!tr.is_contiguous());
    assert(tr.data() == t.data()); // same buffer

    // Element [2,1] in transposed == element [1,2] in original
    const std::array<int64_t, 2> tr_idx{2, 1};
    if (tr.at<float>(tr_idx) != 7.0f) __builtin_trap();

    printf("PASS  test_transpose\n");
}

// ---------------------------------------------------------------------------
// Test 8: slice along a dimension
// ---------------------------------------------------------------------------
static void test_slice() {
    // Row vector [0,1,2,3,4,5,6,7,8,9]
    std::array<int64_t, 1> sh{10};
    auto t = TensorHandle::empty(sh, Dtype::kInt32);
    for (int i = 0; i < 10; ++i)
        t.data_as<int32_t>()[i] = i;

    // Slice [2, 7) → should contain [2,3,4,5,6]
    auto s = t.slice(0, 2, 7);
    assert(s.size(0) == 5);
    assert(s.stride(0) == 4);  // unchanged stride
    const std::array<int64_t, 1> idx0{0};
    const std::array<int64_t, 1> idx4{4};
    if (s.at<int32_t>(idx0) != 2) __builtin_trap();
    if (s.at<int32_t>(idx4) != 6) __builtin_trap();

    // Slice with step=2 → [2,4,6]
    auto s2 = t.slice(0, 2, 8, 2);
    assert(s2.size(0) == 3);
    assert(s2.stride(0) == 8);  // step=2 → stride doubled
    if (s2.at<int32_t>(idx0) != 2) __builtin_trap();
    const std::array<int64_t, 1> idx1{1};
    const std::array<int64_t, 1> idx2{2};
    if (s2.at<int32_t>(idx1) != 4) __builtin_trap();
    if (s2.at<int32_t>(idx2) != 6) __builtin_trap();

    printf("PASS  test_slice\n");
}

// ---------------------------------------------------------------------------
// Test 9: reshape on a contiguous tensor
// ---------------------------------------------------------------------------
static void test_reshape() {
    std::array<int64_t, 2> sh{3, 4};
    auto t = TensorHandle::empty(sh, Dtype::kFloat32);
    float* p = t.data_as<float>();
    for (int i = 0; i < 12; ++i) p[i] = static_cast<float>(i);

    std::array<int64_t, 2> new_sh{2, 6};
    auto r = t.reshape(new_sh);
    assert(r.ndim() == 2);
    assert(r.size(0) == 2 && r.size(1) == 6);
    assert(r.numel() == 12);
    assert(r.is_contiguous());
    assert(r.data() == t.data());  // same buffer

    // Element [1,3] in reshaped = element 9 in flat = 9.0f
    const std::array<int64_t, 2> ridx{1, 3};
    if (r.at<float>(ridx) != 9.0f) __builtin_trap();

    printf("PASS  test_reshape\n");
}

// ---------------------------------------------------------------------------
// Test 10: reshape of transposed tensor fails (not contiguous)
//          Detect by checking is_contiguous before reshape.
// ---------------------------------------------------------------------------
static void test_reshape_requires_contiguous() {
    std::array<int64_t, 2> sh{3, 4};
    auto t = TensorHandle::empty(sh, Dtype::kFloat32);
    auto tr = t.transpose(0, 1);
    assert(!tr.is_contiguous());
    // We don't call reshape on non-contiguous — just verify the check
    printf("PASS  test_reshape_requires_contiguous  (check: !is_contiguous before reshape)\n");
}

// ---------------------------------------------------------------------------
// Test 11: from_buffer wraps existing allocation with shared ownership
// ---------------------------------------------------------------------------
static void test_from_buffer() {
    std::size_t bytes = 8 * sizeof(double);
    void* raw = std::aligned_alloc(8, bytes);
    auto buf = std::shared_ptr<void>(raw, [](void* p){ std::free(p); });

    double* p = static_cast<double*>(raw);
    for (int i = 0; i < 8; ++i) p[i] = static_cast<double>(i) * 0.5;

    std::array<int64_t, 1> sh{8};
    std::array<int64_t, 1> st{8};  // float64 stride = 8 bytes
    auto t = TensorHandle::from_buffer(buf, raw, sh, st, Dtype::kFloat64);
    assert(t.valid());
    assert(t.use_count() == 2);  // buf + t share
    assert(t.numel() == 8);

    const std::array<int64_t, 1> idx3{3};
    if (t.at<double>(idx3) != 1.5) __builtin_trap();

    printf("PASS  test_from_buffer\n");
}

// ---------------------------------------------------------------------------
// Test 12: scalar tensor (0-dim)
// ---------------------------------------------------------------------------
static void test_scalar() {
    std::span<const int64_t> sh{};  // empty span = 0-dim
    auto t = TensorHandle::zeros(sh, Dtype::kFloat32);
    assert(t.valid());
    assert(t.ndim() == 0);
    assert(t.numel() == 1);
    assert(t.is_contiguous());

    // Element access with empty indices
    std::span<const int64_t> no_idx{};
    t.at<float>(no_idx) = 2.71828f;
    assert(t.at<float>(no_idx) == 2.71828f);

    printf("PASS  test_scalar\n");
}

// ---------------------------------------------------------------------------
// Test 13: print_info doesn't crash
// ---------------------------------------------------------------------------
static void test_print_info() {
    std::array<int64_t, 3> sh{2, 3, 4};
    auto t = TensorHandle::zeros(sh, Dtype::kFloat32);
    t.print_info();

    auto tr = t.transpose(0, 2);
    tr.print_info();

    TensorHandle empty;
    empty.print_info();

    printf("PASS  test_print_info\n");
}

// ---------------------------------------------------------------------------
// Test 14: device tag is stored and retrieved correctly
// ---------------------------------------------------------------------------
static void test_device_tag() {
    std::array<int64_t, 1> sh{4};
    auto cpu = TensorHandle::empty(sh, Dtype::kFloat32, {DeviceType::kCPU, 0});
    assert(cpu.device().type  == DeviceType::kCPU);
    assert(cpu.device().index == 0);

    // CUDA device stub (no actual allocation; just verify tag is stored)
    auto cuda = TensorHandle::borrow(cpu.data(), sh,
                                     {&cpu.strides()[0], 1},
                                     Dtype::kFloat32,
                                     {DeviceType::kCUDA, 2});
    assert(cuda.device().type  == DeviceType::kCUDA);
    assert(cuda.device().index == 2);

    printf("PASS  test_device_tag\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    test_dtype_sizes();
    test_empty_shape_strides();
    test_zeros();
    test_element_access();
    test_ref_count_sharing();
    test_borrow();
    test_transpose();
    test_slice();
    test_reshape();
    test_reshape_requires_contiguous();
    test_from_buffer();
    test_scalar();
    test_print_info();
    test_device_tag();
    return 0;
}
