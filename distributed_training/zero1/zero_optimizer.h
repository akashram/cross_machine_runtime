#pragma once
#include <vector>
// TODO: implement on multi-GPU (p4d.24xlarge)

// ZeRO Stage 1: shard optimizer states (momentum + variance) across data-parallel ranks.
// Each rank owns optimizer state for (params / world_size) parameters.
class ZeroStage1Optimizer {
public:
    ZeroStage1Optimizer(std::vector<void*> params, int rank, int world_size,
                         float lr = 1e-4f, float beta1 = 0.9f, float beta2 = 0.999f);
    void step(const std::vector<void*>& grads);
    void zero_grad();
};
