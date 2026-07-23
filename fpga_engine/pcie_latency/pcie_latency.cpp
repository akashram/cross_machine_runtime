// pcie_latency.cpp — decompose PCIe latency into its component costs
//
// "How long does one kernel invocation take" bundles together at least
// four physically distinct costs: the BAR write that rings the doorbell,
// DMA descriptor setup, the data transfer itself, and the
// interrupt/poll overhead of noticing completion. Measuring only the
// bundled total (as dma_orchestration.cpp's two variants do) can't say
// which of these actually dominates. This file measures each in isolation.
//
// TODO: run on F1 with XRT + a real xclbin. Untested — no XRT/FPGA
// available locally, so none of the component latencies below are known.
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_ip.h"

using clock_type = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

namespace {
double mean_ns(const std::vector<ns>& samples) {
    long long total = 0;
    for (auto s : samples) total += s.count();
    return static_cast<double>(total) / samples.size();
}
}

// Component 1: BAR write (doorbell) latency in isolation. xrt::ip's
// write_register() is XRT's documented API for a single MMIO register
// write to a kernel's control BAR — no DMA, no interrupt, just the write
// itself hitting the PCIe endpoint. Repeated many times and averaged to
// smooth out occasional scheduling jitter on the host side.
double measure_bar_write_ns(xrt::device& device, const xrt::uuid& uuid,
                             const std::string& ip_name, int iters) {
    xrt::ip ip(device, uuid, ip_name);
    std::vector<ns> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clock_type::now();
        ip.write_register(/*offset=*/0x0, /*value=*/0x1);
        samples.push_back(std::chrono::duration_cast<ns>(clock_type::now() - t0));
    }
    return mean_ns(samples);
}

// Component 2: DMA descriptor processing + data transfer, decomposed by
// linear regression across buffer sizes. bo.sync() time is modeled as
// descriptor_overhead + size / bandwidth; measuring at several sizes and
// fitting a line separates the fixed per-transfer cost (intercept) from
// the size-dependent cost (slope = 1/bandwidth), rather than assuming
// which one dominates.
struct DmaFit {
    double descriptor_overhead_ns;
    double bandwidth_bytes_per_ns;
};

DmaFit measure_dma_transfer(xrt::device& device, xrt::kernel& kernel, int iters_per_size) {
    const std::vector<size_t> sizes_bytes = {
        4 * 1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024
    };

    std::vector<double> xs, ys; // xs = size (bytes), ys = mean transfer time (ns)
    for (size_t size : sizes_bytes) {
        xrt::bo bo(device, size, kernel.group_id(0));
        std::vector<ns> samples;
        samples.reserve(iters_per_size);
        for (int i = 0; i < iters_per_size; ++i) {
            auto t0 = clock_type::now();
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            samples.push_back(std::chrono::duration_cast<ns>(clock_type::now() - t0));
        }
        xs.push_back(static_cast<double>(size));
        ys.push_back(mean_ns(samples));
    }

    // Ordinary least squares fit: y = intercept + slope * x
    double n = static_cast<double>(xs.size());
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        sum_x += xs[i]; sum_y += ys[i];
        sum_xx += xs[i] * xs[i]; sum_xy += xs[i] * ys[i];
    }
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;

    return DmaFit{intercept, 1.0 / slope};
}

// Component 3: interrupt vs. poll dispatch overhead, isolated from any
// real compute by running the cheapest possible kernel invocation
// (a kernel that does nothing but immediately signal completion) and
// timing only the wait, not the work.
double measure_interrupt_wait_ns(xrt::kernel& kernel, int iters) {
    std::vector<ns> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto run = kernel();
        auto t0 = clock_type::now();
        run.wait();
        samples.push_back(std::chrono::duration_cast<ns>(clock_type::now() - t0));
    }
    return mean_ns(samples);
}

double measure_poll_wait_ns(xrt::kernel& kernel, std::chrono::nanoseconds poll_interval, int iters) {
    std::vector<ns> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto run = kernel();
        auto t0 = clock_type::now();
        while (run.state() != ERT_CMD_STATE_COMPLETED) {
            std::this_thread::sleep_for(poll_interval);
        }
        samples.push_back(std::chrono::duration_cast<ns>(clock_type::now() - t0));
    }
    return mean_ns(samples);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <xclbin> <kernel_name>\n", argv[0]);
        return 1;
    }

    xrt::device device(0);
    auto uuid = device.load_xclbin(argv[1]);
    xrt::kernel kernel(device, uuid, argv[2]);

    double bar_ns = measure_bar_write_ns(device, uuid, argv[2], 1000);
    DmaFit dma = measure_dma_transfer(device, kernel, 50);
    double irq_ns = measure_interrupt_wait_ns(kernel, 200);
    double poll_ns = measure_poll_wait_ns(kernel, std::chrono::microseconds(10), 200);

    std::printf("BAR write (doorbell):        %.1f ns\n", bar_ns);
    std::printf("DMA descriptor overhead:     %.1f ns (fixed, per transfer)\n",
                dma.descriptor_overhead_ns);
    std::printf("DMA bandwidth:                %.3f GB/s\n",
                dma.bandwidth_bytes_per_ns);
    std::printf("Interrupt dispatch overhead: %.1f ns\n", irq_ns);
    std::printf("Poll dispatch overhead:      %.1f ns (10us interval)\n", poll_ns);
    return 0;
}
