// cutlass_gemm.cu — a CUTLASS-template GEMM instance, benchmarked against
// gpu_engine/kernels/gemm.cuh's hand-written tiled GEMM (step 9) and
// gpu_engine/hopper/wgmma.cuh's hand-written WGMMA kernel (step 19).
//
// CUTLASS (NVIDIA's open-source template library, github.com/NVIDIA/cutlass)
// composes a GEMM out of the same conceptual pieces gemm.cuh and wgmma.cuh
// hand-wrote: a threadblock-level tile, a warp-level tile, an instruction-
// level MMA op, and an epilogue. The difference is that here they are
// template parameters chosen at compile time, not code written by hand —
// this file *instantiates* a GEMM rather than *implementing* one.
//
// Instance chosen: SM80 (Ampere) Tensor Core GEMM, FP16 inputs / FP32
// accumulate — the same dtype pairing gemm_wmma uses, so the comparison in
// the README is apples-to-apples against step 9's WMMA variant. A second,
// commented-out instantiation shows the SM90 (Hopper) collective-builder
// API that would target the same WGMMA instructions wgmma.cuh hand-writes
// in PTX; it is left commented rather than active because this repo has no
// H100 to compile/verify it against (same hardware gate as gpu_engine/hopper),
// and CUTLASS's SM90 collective API is a different, larger surface than the
// SM80 path below — instantiating it without ever building it would just be
// a second unverified guess stacked on the first.
//
// Status: HARDWARE-GATED, UNRUN. Needs a CUDA GPU (SM80+ for the Tensor Core
// path below) and the CUTLASS headers (fetched by CMakeLists.txt via
// FetchContent — header-only, no separate build/link step).

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));          \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// -----------------------------------------------------------------------
// The GEMM "instance": every knob gemm_tiled<TILE> and gemm_wmma hard-code
// into the kernel body is here a template parameter instead.
//
//   ElementA/B    = cutlass::half_t   — same dtype gemm_wmma requires
//   ElementC      = float             — same accumulate dtype as gemm_wmma
//   LayoutA/B/C   = RowMajor          — matches gemm.cuh's row-major convention
//   OpClass       = TensorOp          — routes through Tensor Core MMA,
//                                       CUTLASS's equivalent of gemm_wmma's
//                                       explicit wmma::mma_sync calls
//   ArchTag       = Sm80              — Ampere; A100-class (p3/p4d instances
//                                       already used for steps 1-24)
//   ThreadblockShape / WarpShape / InstructionShape
//                 = the threadblock/warp/instruction tiling hierarchy —
//                   the templated analogue of gemm_tiled<TILE>'s __shared__
//                   As[TILE][TILE]/Bs[TILE][TILE] tiles, chosen here from
//                   CUTLASS's documented well-tuned presets for SM80 FP16
//                   rather than hand-tuned the way gemm.cuh's README
//                   documents picking TILE=16 vs TILE=32 by hand.
// -----------------------------------------------------------------------

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,   // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,     // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,      // Tensor Core instruction shape
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3   // pipeline stages — CUTLASS's equivalent of gemm.py's num_stages
>;

// SM90 (Hopper) sketch — NOT compiled, no H100 to verify against. Left as
// a real, plausible instantiation sketch rather than omitted, matching how
// gpu_engine/hopper documents its own WGMMA path: the collective-builder
// API composes CollectiveMainloop (warpgroup-level wgmma.mma_async, same
// instruction wgmma.cuh hand-writes in PTX) with CollectiveEpilogue via
// cutlass::gemm::collective::CollectiveBuilder, a materially different and
// newer API surface than the SM80 device::Gemm path above.
//
// using CollectiveMainloopSm90 = typename cutlass::gemm::collective::CollectiveBuilder<
//     cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
//     cutlass::half_t, LayoutA, 8,
//     cutlass::half_t, LayoutB, 8,
//     float,
//     cute::Shape<cute::_128, cute::_256, cute::_64>,  // CTA tile
//     cute::Shape<cute::_1, cute::_1, cute::_1>,        // cluster shape
//     cutlass::gemm::collective::StageCountAuto,
//     cutlass::gemm::collective::KernelScheduleAuto
// >::CollectiveOp;

static bool run_and_verify(int M, int N, int K) {
    cutlass::HostTensor<ElementA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> C({M, N});
    cutlass::HostTensor<ElementC, LayoutC> C_ref({M, N});

    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 42, ElementA(2), ElementA(-2));
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 43, ElementB(2), ElementB(-2));
    cutlass::reference::host::TensorFill(C.host_view(), ElementC(0));

    A.sync_device();
    B.sync_device();
    C.sync_device();

    Gemm gemm_op;
    typename Gemm::Arguments args(
        {M, N, K},
        A.device_ref(), B.device_ref(), C.device_ref(), C.device_ref(),
        {ElementAccumulator(1), ElementAccumulator(0)});

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS kernel cannot implement this problem size: %s\n",
                cutlass::cutlassGetStatusString(status));
        return false;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS init failed: %s\n", cutlass::cutlassGetStatusString(status));
        return false;
    }

    // Warmup + timed loop — same cudaEvent_t pattern as gemm_bench.cu's
    // time_kernel_ms, so TFLOPS numbers land in comparable units.
    for (int i = 0; i < 5; ++i) gemm_op();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    constexpr int iters = 20;
    for (int i = 0; i < iters; ++i) gemm_op();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    double tflops = 2.0 * M * N * K / (ms / 1e3) / 1e12;
    printf("CUTLASS SM80 TensorOp GEMM  M=N=K=%d  %.3f ms  %.2f TFLOPS\n", M, ms, tflops);

    // Correctness: CUTLASS's own reference GEMM, same role gemm_bench.cu's
    // verify() against cuBLAS plays for the hand-written variants.
    C.sync_host();
    cutlass::reference::host::Gemm<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
        ElementAccumulator, ElementAccumulator>
        reference_gemm;
    reference_gemm({M, N, K}, ElementAccumulator(1), A.host_ref(), B.host_ref(),
                    ElementAccumulator(0), C_ref.host_ref());

    bool passed = cutlass::reference::host::TensorEquals(C.host_view(), C_ref.host_view());
    printf("  correctness vs. CUTLASS host reference: %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

int main() {
    bool ok = true;
    for (int sz : {512, 1024, 2048, 4096}) {
        ok &= run_and_verify(sz, sz, sz);
    }
    return ok ? 0 : 1;
}
