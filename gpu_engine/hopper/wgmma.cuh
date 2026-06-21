#pragma once
// Hopper WGMMA (Warpgroup Matrix Multiply-Accumulate) — sm_90 only.
//
// WGMMA is the Hopper successor to WMMA (sm_70) and HMMA (sm_80 TF32).
// Key differences from WMMA:
//   - Warpgroup scope: 4 warps (128 threads) collaborate on one MMA.
//   - Much larger tiles: e.g. 64×256×16 for BF16, vs WMMA's 16×16×16.
//   - Async execution: wgmma.mma_async lets the SM overlap MMA with data fetch.
//   - Accumulator in registers: distributed across all 128 threads.
//
// The PTX instruction:
//   wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
//
// In CUDA 12.0, WGMMA is accessible via:
//   1. Inline PTX (most flexible, exact control)
//   2. cuda::experimental::wgmma (limited availability in early CUDA 12.x)
//   3. CUTLASS 3.x (production-ready abstraction)
//
// This header implements a PTX-level wrapper for the m64n256k16 shape.
// Adjust M/N/K template params for other supported shapes.
//
// Supported shapes (BF16 input, FP32 accumulator):
//   m=64, n={8,16,24,32,40,48,56,64,80,96,112,128,144,160,176,192,208,224,240,256}
//   k=16
//
// Thread layout: 128 threads, each owns (M*N)/128 elements of the accumulator.
// For m=64, n=256: 64*256/128 = 128 accumulator floats per thread.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900

// -----------------------------------------------------------------------
// Descriptor helpers — shared memory descriptor for WGMMA input matrices.
//
// WGMMA reads A from shared memory and B from either shared or global memory.
// The descriptor encodes layout info the hardware needs.
//
// For A in shared memory with row-major layout and no swizzle:
//   desc = smem_ptr_as_u64 | (stride_64b << 16) | (leading_offset << 32)
// This layout matches CUTLASS's SmemDescriptor for non-swizzled shared memory.
// -----------------------------------------------------------------------

__device__ __forceinline__
uint64_t make_smem_desc(const void* smem_ptr, int stride_bytes) {
    // Shared memory address (in bytes, assumed 128-byte aligned)
    uint64_t addr  = reinterpret_cast<uint64_t>(__cvta_generic_to_shared(smem_ptr));
    // stride_bytes / 64 (units of 8 × __bfloat16 elements = 16 bytes)... actually:
    // The leading_byte_offset and stride are in units of 8 bytes for BF16.
    uint64_t stride = (uint64_t)(stride_bytes / 8);  // in units of 8 bytes
    // Descriptor format (simplified for no-swizzle):
    //   [13:0]  = start address >> 4 (16-byte aligned)
    //   [29:16] = stride in 8-byte units
    //   [45:32] = leading dimension offset (0 for row-major)
    uint64_t desc = ((addr >> 4) & 0x3FFF)
                  | (stride << 16)
                  | (0ULL  << 32);  // no leading offset
    return desc;
}

// -----------------------------------------------------------------------
// m64n256k16 WGMMA: each warpgroup (128 threads) computes a 64×256 tile.
// A is 64×16 BF16 in shared memory; B is 16×256 BF16 in shared memory.
// Accumulator `acc` has 128 FP32 values per thread (distributed across 128 threads).
// -----------------------------------------------------------------------

constexpr int WGMMA_M = 64, WGMMA_N = 256, WGMMA_K = 16;
constexpr int WGMMA_ACC_PER_THREAD = WGMMA_M * WGMMA_N / 128;  // = 128

template <int M = WGMMA_M, int N = WGMMA_N, int K = WGMMA_K>
__device__ __forceinline__
void wgmma_mma_async_bf16_f32(
    float           acc[WGMMA_ACC_PER_THREAD],  // in-register accumulator
    const __bfloat16* a_smem,                    // [M x K] in shared memory
    const __bfloat16* b_smem)                    // [K x N] in shared memory
{
    static_assert(M == 64 && N == 256 && K == 16,
                  "This instantiation only supports m64n256k16");

    uint64_t desc_a = make_smem_desc(a_smem, M * sizeof(__bfloat16));
    uint64_t desc_b = make_smem_desc(b_smem, K * sizeof(__bfloat16));

    // Issue the wgmma.mma_async instruction via PTX.
    // The accumulator register file layout for m64n256k16 has 64 FP32 pairs (128 regs).
    // The PTX instruction updates all acc registers in the warpgroup collectively.
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        // wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "  {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
        "   %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29,"
        "   %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43,"
        "   %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57,"
        "   %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71,"
        "   %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85,"
        "   %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99,"
        "   %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111,"
        "   %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123,"
        "   %124, %125, %126, %127},\n"
        "  %128, %129, 1, 1, 1, 0, 0;\n"
        "}"
        : "+f"(acc[  0]), "+f"(acc[  1]), "+f"(acc[  2]), "+f"(acc[  3]),
          "+f"(acc[  4]), "+f"(acc[  5]), "+f"(acc[  6]), "+f"(acc[  7]),
          "+f"(acc[  8]), "+f"(acc[  9]), "+f"(acc[ 10]), "+f"(acc[ 11]),
          "+f"(acc[ 12]), "+f"(acc[ 13]), "+f"(acc[ 14]), "+f"(acc[ 15]),
          "+f"(acc[ 16]), "+f"(acc[ 17]), "+f"(acc[ 18]), "+f"(acc[ 19]),
          "+f"(acc[ 20]), "+f"(acc[ 21]), "+f"(acc[ 22]), "+f"(acc[ 23]),
          "+f"(acc[ 24]), "+f"(acc[ 25]), "+f"(acc[ 26]), "+f"(acc[ 27]),
          "+f"(acc[ 28]), "+f"(acc[ 29]), "+f"(acc[ 30]), "+f"(acc[ 31]),
          "+f"(acc[ 32]), "+f"(acc[ 33]), "+f"(acc[ 34]), "+f"(acc[ 35]),
          "+f"(acc[ 36]), "+f"(acc[ 37]), "+f"(acc[ 38]), "+f"(acc[ 39]),
          "+f"(acc[ 40]), "+f"(acc[ 41]), "+f"(acc[ 42]), "+f"(acc[ 43]),
          "+f"(acc[ 44]), "+f"(acc[ 45]), "+f"(acc[ 46]), "+f"(acc[ 47]),
          "+f"(acc[ 48]), "+f"(acc[ 49]), "+f"(acc[ 50]), "+f"(acc[ 51]),
          "+f"(acc[ 52]), "+f"(acc[ 53]), "+f"(acc[ 54]), "+f"(acc[ 55]),
          "+f"(acc[ 56]), "+f"(acc[ 57]), "+f"(acc[ 58]), "+f"(acc[ 59]),
          "+f"(acc[ 60]), "+f"(acc[ 61]), "+f"(acc[ 62]), "+f"(acc[ 63]),
          "+f"(acc[ 64]), "+f"(acc[ 65]), "+f"(acc[ 66]), "+f"(acc[ 67]),
          "+f"(acc[ 68]), "+f"(acc[ 69]), "+f"(acc[ 70]), "+f"(acc[ 71]),
          "+f"(acc[ 72]), "+f"(acc[ 73]), "+f"(acc[ 74]), "+f"(acc[ 75]),
          "+f"(acc[ 76]), "+f"(acc[ 77]), "+f"(acc[ 78]), "+f"(acc[ 79]),
          "+f"(acc[ 80]), "+f"(acc[ 81]), "+f"(acc[ 82]), "+f"(acc[ 83]),
          "+f"(acc[ 84]), "+f"(acc[ 85]), "+f"(acc[ 86]), "+f"(acc[ 87]),
          "+f"(acc[ 88]), "+f"(acc[ 89]), "+f"(acc[ 90]), "+f"(acc[ 91]),
          "+f"(acc[ 92]), "+f"(acc[ 93]), "+f"(acc[ 94]), "+f"(acc[ 95]),
          "+f"(acc[ 96]), "+f"(acc[ 97]), "+f"(acc[ 98]), "+f"(acc[ 99]),
          "+f"(acc[100]), "+f"(acc[101]), "+f"(acc[102]), "+f"(acc[103]),
          "+f"(acc[104]), "+f"(acc[105]), "+f"(acc[106]), "+f"(acc[107]),
          "+f"(acc[108]), "+f"(acc[109]), "+f"(acc[110]), "+f"(acc[111]),
          "+f"(acc[112]), "+f"(acc[113]), "+f"(acc[114]), "+f"(acc[115]),
          "+f"(acc[116]), "+f"(acc[117]), "+f"(acc[118]), "+f"(acc[119]),
          "+f"(acc[120]), "+f"(acc[121]), "+f"(acc[122]), "+f"(acc[123]),
          "+f"(acc[124]), "+f"(acc[125]), "+f"(acc[126]), "+f"(acc[127])
        : "l"(desc_a), "l"(desc_b)
    );
}

// Commit wgmma operations: required after all wgmma.mma_async to ensure
// the results are visible in `acc` registers before use.
__device__ __forceinline__
void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" :::);
}

// Wait for committed groups to complete.
// N=0: wait for all outstanding groups; N>0: leave N groups outstanding.
template<int N = 0>
__device__ __forceinline__
void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) :);
}

#else  // non-Hopper fallback

constexpr int WGMMA_ACC_PER_THREAD = 128;

template <int M = 64, int N = 256, int K = 16>
__device__ __forceinline__
void wgmma_mma_async_bf16_f32(float*, const void*, const void*) {}

__device__ __forceinline__ void wgmma_commit_group() {}
template<int N = 0>
__device__ __forceinline__ void wgmma_wait_group() {}

#endif  // __CUDA_ARCH__ >= 900
