// flash_attn_bench.cu — compare naive / cuDNN / Flash Attention latency and memory
// TODO: run on GPU hardware

#include "flash_attn_fwd.cuh"
#include "flash_attn_bwd.cuh"
#include <cudnn.h>
#include <cstdio>

int main() {
    // TODO: implement on GPU hardware
    // Test config: B=16, H=12, N=2048, D=64 (BERT-large-ish)
    //
    // For each variant (naive, cuDNN, flash):
    //   - measure forward latency (ms) via cudaEvent
    //   - measure peak HBM usage via cudaMemGetInfo before/after
    //   - verify outputs match to within 1e-2 (FP16 tolerance)
    //
    // For backward:
    //   - compare dQ/dK/dV against PyTorch autograd reference
    //   - measure backward latency and memory
    //
    // Goal: flash_attn_fwd should beat cuDNN on memory by ~4x at seq_len=2048

    printf("flash_attn_bench: STUB — run on GPU hardware (A100 recommended)\n");
    return 0;
}
