// dma_orchestration.cpp — host -> FPGA -> host DMA via XRT, interrupt vs poll
//
// Real XRT native C++ API usage (xrt::device / xrt::bo / xrt::kernel /
// xrt::run), not a stub — the DMA orchestration logic itself (buffer
// allocation, sync direction, completion wait strategy) doesn't depend on
// which kernel is running underneath, so it's written once against any
// .xclbin with a single m_axi buffer in/out kernel (dot_product works).
//
// TODO: run on F1 with a real xclbin loaded and XRT installed. Untested —
// no XRT runtime or FPGA card available locally, so none of the latency
// numbers below are measured yet.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

using clock_type = std::chrono::steady_clock;

namespace {

double ms_since(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

// Shared setup: open device, load xclbin, allocate host<->device buffers
// for a kernel with (const float* a, const float* b, float* result, int n)
// — dot_product's signature. Returns everything the two completion-wait
// variants below need, so the DMA/compute steps themselves are identical
// and only the wait strategy differs between interrupt_variant/poll_variant.
struct KernelContext {
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo bo_a, bo_b, bo_result;
    int n;
};

KernelContext setup(const std::string& xclbin_path, const std::string& kernel_name, int n) {
    xrt::device device(0); // first FPGA device enumerated by XRT
    auto uuid = device.load_xclbin(xclbin_path);
    xrt::kernel kernel(device, uuid, kernel_name);

    xrt::bo bo_a(device, n * sizeof(float), kernel.group_id(0));
    xrt::bo bo_b(device, n * sizeof(float), kernel.group_id(1));
    xrt::bo bo_result(device, sizeof(float), kernel.group_id(2));

    return KernelContext{std::move(device), std::move(kernel),
                          std::move(bo_a), std::move(bo_b), std::move(bo_result), n};
}

void fill_inputs(KernelContext& ctx, const std::vector<float>& a, const std::vector<float>& b) {
    auto* a_map = ctx.bo_a.map<float*>();
    auto* b_map = ctx.bo_b.map<float*>();
    std::copy(a.begin(), a.end(), a_map);
    std::copy(b.begin(), b.end(), b_map);
    ctx.bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    ctx.bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

} // namespace

// Interrupt-driven completion: xrt::run::wait() blocks the calling thread
// on the kernel's completion interrupt via XRT's internal event mechanism
// (ultimately a driver-level wait on an eventfd/uio interrupt line) — no
// host CPU spent spinning, but each wait() pays interrupt latency
// (context switch + ISR dispatch) before the host observes completion.
double run_interrupt_variant(const std::string& xclbin_path, const std::string& kernel_name,
                              const std::vector<float>& a, const std::vector<float>& b) {
    KernelContext ctx = setup(xclbin_path, kernel_name, static_cast<int>(a.size()));

    auto start = clock_type::now();
    fill_inputs(ctx, a, b);

    auto run = ctx.kernel(ctx.bo_a, ctx.bo_b, ctx.bo_result, ctx.n);
    run.wait(); // interrupt-driven block until kernel-done IRQ fires

    ctx.bo_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    double elapsed_ms = ms_since(start);

    float result = *ctx.bo_result.map<float*>();
    std::printf("interrupt variant: result=%f elapsed=%.4f ms\n", result, elapsed_ms);
    return elapsed_ms;
}

// Polling completion: repeatedly checks run state instead of blocking on
// an interrupt. Burns host CPU on the polling thread, but observes
// completion at the next poll tick rather than waiting for interrupt
// dispatch — expected to win on latency for short-running kernels where
// interrupt overhead is a larger fraction of total time, and lose on CPU
// utilization for long-running ones.
double run_poll_variant(const std::string& xclbin_path, const std::string& kernel_name,
                         const std::vector<float>& a, const std::vector<float>& b,
                         std::chrono::microseconds poll_interval) {
    KernelContext ctx = setup(xclbin_path, kernel_name, static_cast<int>(a.size()));

    auto start = clock_type::now();
    fill_inputs(ctx, a, b);

    auto run = ctx.kernel(ctx.bo_a, ctx.bo_b, ctx.bo_result, ctx.n);
    while (run.state() != ERT_CMD_STATE_COMPLETED) {
        std::this_thread::sleep_for(poll_interval);
    }

    ctx.bo_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    double elapsed_ms = ms_since(start);

    float result = *ctx.bo_result.map<float*>();
    std::printf("poll variant: result=%f elapsed=%.4f ms (poll_interval=%lldus)\n",
                result, elapsed_ms, static_cast<long long>(poll_interval.count()));
    return elapsed_ms;
}
