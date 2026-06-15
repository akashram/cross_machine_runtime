#include <cstdio>
#include <cuda_runtime.h>

static void check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d — %s\n",
                     file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CHECK(x) check((x), __FILE__, __LINE__)

int main() {
    int driver_version = 0, runtime_version = 0;
    CHECK(cudaDriverGetVersion(&driver_version));
    CHECK(cudaRuntimeGetVersion(&runtime_version));
    std::printf("CUDA driver:  %d.%d\n",
                driver_version / 1000, (driver_version % 100) / 10);
    std::printf("CUDA runtime: %d.%d\n\n",
                runtime_version / 1000, (runtime_version % 100) / 10);

    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    std::printf("Found %d CUDA device(s)\n\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        CHECK(cudaGetDeviceProperties(&p, i));

        // Theoretical peak memory bandwidth (GB/s):
        // (memoryClockRate in kHz) * 2 (DDR) * (memoryBusWidth bits / 8) / 1e6
        double bw_gbs = (static_cast<double>(p.memoryClockRate) * 1e3 * 2.0
                         * (p.memoryBusWidth / 8.0)) / 1e9;

        std::printf("Device %d: %s\n", i, p.name);
        std::printf("  Compute capability:    %d.%d (sm_%d%d)\n",
                    p.major, p.minor, p.major, p.minor);
        std::printf("  SMs:                   %d\n", p.multiProcessorCount);
        std::printf("  CUDA cores (est.):     %d\n",
                    p.multiProcessorCount * 128);  // approximate; varies by arch
        std::printf("  Global memory:         %.1f GB\n",
                    static_cast<double>(p.totalGlobalMem) / (1ull << 30));
        std::printf("  Peak mem bandwidth:    %.1f GB/s\n", bw_gbs);
        std::printf("  L2 cache:              %.1f MB\n",
                    static_cast<double>(p.l2CacheSize) / (1 << 20));
        std::printf("  Shared mem / block:    %zu KB\n",
                    p.sharedMemPerBlock / 1024);
        std::printf("  Registers / block:     %d\n", p.regsPerBlock);
        std::printf("  Warp size:             %d\n", p.warpSize);
        std::printf("  Max threads / block:   %d\n", p.maxThreadsPerBlock);
        std::printf("  Max threads / SM:      %d\n", p.maxThreadsPerMultiProcessor);
        std::printf("  Max warps / SM:        %d\n",
                    p.maxThreadsPerMultiProcessor / p.warpSize);
        std::printf("  Clock rate:            %.0f MHz\n",
                    static_cast<double>(p.clockRate) / 1e3);
        std::printf("  ECC enabled:           %s\n",
                    p.ECCEnabled ? "yes" : "no");
        std::printf("  Unified addressing:    %s\n",
                    p.unifiedAddressing ? "yes" : "no");
        std::printf("  Peer access capable:   %s\n",
                    p.directManagedMemAccessFromHost ? "yes" : "no");
        std::printf("\n");
    }
    return 0;
}
