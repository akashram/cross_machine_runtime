// power_bench.cu — measure per-kernel power draw and thermal throttling
// TODO: run on GPU hardware
#include "power_monitor.h"
#include <cstdio>

int main() {
    // TODO: implement on GPU hardware
    // For each kernel (idle, add, gemm_wmma, flash_attn):
    //   monitor.start();
    //   run kernel for 10 seconds
    //   PowerReport r = monitor.stop();
    //   print r.avg_power_w, r.peak_power_w, r.max_temp_c, r.throttled
    printf("power_bench: STUB — run on Linux GPU hardware\n");
    return 0;
}
