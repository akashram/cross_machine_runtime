#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

static void check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CHECK(x) check((x), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// cudaEvent-based bandwidth measurement
// cudaEventElapsedTime measures GPU-side elapsed time for the async work
// submitted between the two Record calls. For synchronous cudaMemcpy this
// captures only the DMA transfer time, not driver overhead.
// ---------------------------------------------------------------------------
struct Stats { double mean, p50, p99, best; };

static Stats bench_bw(void* dst, const void* src, std::size_t bytes,
                      cudaMemcpyKind kind, int iters = 30) {
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    // 3 warmup transfers so caches/TLBs are in steady state
    for (int i = 0; i < 3; ++i)
        CHECK(cudaMemcpy(dst, src, bytes, kind));
    CHECK(cudaDeviceSynchronize());

    std::vector<double> bw;
    bw.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(t0));
        CHECK(cudaMemcpy(dst, src, bytes, kind));
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        bw.push_back(static_cast<double>(bytes) / (ms * 1e-3) / 1e9);
    }

    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));

    std::vector<double> sorted = bw;
    std::sort(sorted.begin(), sorted.end());
    double sum = std::accumulate(bw.begin(), bw.end(), 0.0);
    return {
        sum / iters,
        sorted[iters / 2],
        sorted[iters * 99 / 100],
        sorted.back()
    };
}

static void print_bw_header() {
    std::printf("%-9s  %10s  %10s  %10s  %10s\n",
                "Size", "Mean GB/s", "p50 GB/s", "p99 GB/s", "Best GB/s");
}

static void print_bw_row(const char* label, const Stats& s) {
    std::printf("%-9s  %10.2f  %10.2f  %10.2f  %10.2f\n",
                label, s.mean, s.p50, s.p99, s.best);
}

int main() {
    CHECK(cudaSetDevice(0));

    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));

    double theoretical_bw =
        static_cast<double>(p.memoryClockRate) * 1e3 * 2.0
        * (p.memoryBusWidth / 8.0) / 1e9;

    std::printf("Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    std::printf("Global memory:        %.1f GB\n",
                static_cast<double>(p.totalGlobalMem) / (1ull << 30));
    std::printf("Theoretical HBM BW:   %.1f GB/s\n", theoretical_bw);
    std::printf("PCIe gen/width:       Gen%d x%d (theoretical ~%.0f GB/s per direction)\n\n",
                p.pciBusID > 0 ? 4 : 3,  // rough heuristic
                16,
                16.0);  // PCIe 4.0 x16 = ~32 GB/s total; ~16 per direction

    // Transfer sizes: 1 MB → 512 MB
    static const std::size_t kSizes[] = {
        1ull  << 20,   //   1 MB
        4ull  << 20,   //   4 MB
        16ull << 20,   //  16 MB
        64ull << 20,   //  64 MB
        256ull<< 20,   // 256 MB
        512ull<< 20,   // 512 MB
    };
    static const char* kLabels[] = {
        "1 MB", "4 MB", "16 MB", "64 MB", "256 MB", "512 MB"
    };
    const int kN = static_cast<int>(sizeof(kSizes) / sizeof(kSizes[0]));

    // -----------------------------------------------------------------------
    // cudaMalloc latency
    // Host-side wall-clock time for a single allocation + free.
    // This is driver overhead, not DMA. Expected: ~8–15 µs regardless of size
    // (it's a kernel call to the CUDA driver, not proportional to bytes).
    // -----------------------------------------------------------------------
    std::printf("=== cudaMalloc latency (host wall-clock, N=200) ===\n");
    std::printf("%-9s  %10s  %10s  %10s\n", "Size", "p50 µs", "p99 µs", "mean µs");
    {
        const int N = 200;
        std::size_t lat_sizes[] = {4096, 1<<20, 64<<20, 256<<20};
        const char* lat_labels[] = {"4 KB", "1 MB", "64 MB", "256 MB"};
        for (int s = 0; s < 4; ++s) {
            std::vector<double> us;
            us.reserve(N);
            for (int i = 0; i < N; ++i) {
                void* ptr = nullptr;
                auto t0 = std::chrono::steady_clock::now();
                cudaMalloc(&ptr, lat_sizes[s]);
                auto t1 = std::chrono::steady_clock::now();
                cudaFree(ptr);
                double elapsed =
                    std::chrono::duration<double, std::micro>(t1 - t0).count();
                us.push_back(elapsed);
            }
            std::vector<double> sorted = us;
            std::sort(sorted.begin(), sorted.end());
            double mean = std::accumulate(us.begin(), us.end(), 0.0) / N;
            std::printf("%-9s  %10.1f  %10.1f  %10.1f\n",
                        lat_labels[s],
                        sorted[N / 2],
                        sorted[N * 99 / 100],
                        mean);
        }
    }
    std::printf("\n");

    // -----------------------------------------------------------------------
    // H2D: pageable host → device
    // cudaMemcpy must first copy from pageable memory to a locked staging
    // buffer in the driver, then DMA from that buffer to device. Two copies
    // on the host side before the PCIe transfer starts.
    // -----------------------------------------------------------------------
    std::printf("=== H2D bandwidth: pageable host → device ===\n");
    print_bw_header();
    for (int i = 0; i < kN; ++i) {
        std::size_t bytes = kSizes[i];
        void* host = std::malloc(bytes);
        std::memset(host, 0xab, bytes);
        void* dev = nullptr;
        CHECK(cudaMalloc(&dev, bytes));
        auto s = bench_bw(dev, host, bytes, cudaMemcpyHostToDevice);
        print_bw_row(kLabels[i], s);
        std::free(host);
        cudaFree(dev);
    }
    std::printf("\n");

    // -----------------------------------------------------------------------
    // H2D: pinned host → device
    // No staging copy needed. The DMA engine reads directly from the locked
    // physical pages. Expected: ~2× faster than pageable for large transfers.
    // -----------------------------------------------------------------------
    std::printf("=== H2D bandwidth: pinned host → device ===\n");
    print_bw_header();
    for (int i = 0; i < kN; ++i) {
        std::size_t bytes = kSizes[i];
        void* host = nullptr;
        CHECK(cudaMallocHost(&host, bytes));
        std::memset(host, 0xab, bytes);
        void* dev = nullptr;
        CHECK(cudaMalloc(&dev, bytes));
        auto s = bench_bw(dev, host, bytes, cudaMemcpyHostToDevice);
        print_bw_row(kLabels[i], s);
        cudaFreeHost(host);
        cudaFree(dev);
    }
    std::printf("\n");

    // -----------------------------------------------------------------------
    // D2H: device → pinned host
    // Symmetric to pinned H2D; measures PCIe upload vs download asymmetry.
    // -----------------------------------------------------------------------
    std::printf("=== D2H bandwidth: device → pinned host ===\n");
    print_bw_header();
    for (int i = 0; i < kN; ++i) {
        std::size_t bytes = kSizes[i];
        void* dev = nullptr;
        CHECK(cudaMalloc(&dev, bytes));
        CHECK(cudaMemset(dev, 0, bytes));
        void* host = nullptr;
        CHECK(cudaMallocHost(&host, bytes));
        auto s = bench_bw(host, dev, bytes, cudaMemcpyDeviceToHost);
        print_bw_row(kLabels[i], s);
        cudaFree(dev);
        cudaFreeHost(host);
    }
    std::printf("\n");

    // -----------------------------------------------------------------------
    // D2D: device → device (same GPU)
    // Goes through L2 cache on Ampere+; on older architectures goes through
    // HBM (reads src from HBM, writes dst to HBM — two passes, so ~50% of
    // peak HBM bandwidth). Expected: 150–300 GB/s on T4; ~1.5 TB/s on A100.
    // -----------------------------------------------------------------------
    std::printf("=== D2D bandwidth: device → device (same GPU) ===\n");
    print_bw_header();
    for (int i = 0; i < kN; ++i) {
        std::size_t bytes = kSizes[i];
        void* src = nullptr;
        void* dst = nullptr;
        CHECK(cudaMalloc(&src, bytes));
        CHECK(cudaMalloc(&dst, bytes));
        CHECK(cudaMemset(src, 0, bytes));
        auto s = bench_bw(dst, src, bytes, cudaMemcpyDeviceToDevice);
        print_bw_row(kLabels[i], s);
        cudaFree(src);
        cudaFree(dst);
    }
    std::printf("\n");

    std::printf("Theoretical HBM peak: %.1f GB/s\n", theoretical_bw);
    std::printf("D2D ceiling (2 passes through HBM): ~%.0f GB/s\n",
                theoretical_bw / 2.0);

    return 0;
}
