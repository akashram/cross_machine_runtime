// p2p_bench.cu — direct GPU-to-GPU vs host-staged transfer bandwidth.
//
// On NVLink systems, P2P bandwidth should be ~600 GB/s aggregate (A100 NVLink-3)
// vs ~32 GB/s PCIe 4.0 × 16.  This benchmark makes the gap visible.
//
// If only one GPU is present: reports single-device bandwidth (memcpy within device).
// If multiple GPUs: prints a full peer-access matrix + bandwidth table.

#include "gpu_engine/p2p/p2p.h"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));          \
        exit(1);                                                      \
    }                                                                 \
} while (0)

static void print_device_info(int id) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, id));
    printf("  GPU %d: %s (CC %d.%d, %.0f GB HBM, PCIe bus %02d)\n",
           id, p.name, p.major, p.minor,
           p.totalGlobalMem / 1e9, p.pciBusID);
}

int main() {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    printf("=== Device topology (%d GPU(s)) ===\n", ndev);
    for (int i = 0; i < ndev; ++i) print_device_info(i);
    printf("\n");

    if (ndev < 2) {
        // Single GPU: measure intra-device bandwidth (memcpy within same GPU)
        printf("Only 1 GPU detected — measuring intra-device bandwidth.\n");
        printf("Use p4d.24xlarge (8× A100 NVLink) for meaningful P2P numbers.\n\n");

        const size_t bytes = 256 * 1024 * 1024;  // 256 MB
        void *d_a, *d_b;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int i = 0; i < 20; ++i)
            CUDA_CHECK(cudaMemcpyAsync(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double bw = bytes / (ms / 20.0 / 1e3) / 1e9;
        printf("  Intra-device bandwidth: %.1f GB/s  (%.1f MB in %.3f ms)\n",
               bw, bytes / 1e6, ms / 20.0);

        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        return 0;
    }

    // ---- Peer access matrix -------------------------------------------
    printf("=== Peer access matrix ===\n");
    printf("  src\\dst  ");
    for (int d = 0; d < ndev; ++d) printf("  GPU%d", d);
    printf("\n");
    for (int s = 0; s < ndev; ++s) {
        printf("  GPU%d     ", s);
        for (int d = 0; d < ndev; ++d) {
            if (s == d) { printf("   --"); continue; }
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, s, d));
            printf("  %s", can ? " yes" : "  no");
        }
        printf("\n");
    }
    printf("\n");

    // Enable peer access for all pairs that support it
    for (int s = 0; s < ndev; ++s)
        for (int d = 0; d < ndev; ++d)
            if (s != d) enable_peer_access(s, d);

    // ---- Bandwidth sweep ----------------------------------------------
    const size_t sizes[] = {
        1  * 1024 * 1024,    //   1 MB
        16 * 1024 * 1024,    //  16 MB
        256* 1024 * 1024,    // 256 MB
        1024ULL * 1024 * 1024,  // 1 GB
    };
    const char* size_labels[] = {"1 MB", "16 MB", "256 MB", "1 GB"};

    for (int s = 0; s < ndev; ++s) {
        for (int d = s + 1; d < ndev; ++d) {
            printf("=== GPU%d → GPU%d ===\n", s, d);
            printf("  %-10s  %-16s  %-16s  %s\n",
                   "size", "P2P (GB/s)", "staged (GB/s)", "P2P / staged");
            printf("  %s\n", std::string(58, '-').c_str());

            for (int i = 0; i < 4; ++i) {
                size_t bytes = sizes[i];
                auto p2p    = measure_p2p_bandwidth(s, d, bytes);
                auto staged = measure_host_staged_bandwidth(s, d, bytes);
                printf("  %-10s  %-16.2f  %-16.2f  %.2f×%s\n",
                       size_labels[i],
                       p2p.bandwidth_gbs,
                       staged.bandwidth_gbs,
                       p2p.bandwidth_gbs / staged.bandwidth_gbs,
                       p2p.nvlink ? "  (NVLink)" : "  (PCIe)");
            }
            printf("\n");
        }
    }

    return 0;
}
