#pragma once
// TODO: implement on multi-GPU

// Mixture-of-Experts layer with learned top-k routing.
// Dispatches tokens to expert processes via all-to-all.
class MoeLayer {
public:
    MoeLayer(int num_experts, int expert_hidden_dim, int top_k, int rank, int world_size);

    // Forward: route each token to top_k experts, compute, combine results.
    void forward(const float* input,  // [batch, seq, hidden]
                 float* output,        // [batch, seq, hidden]
                 int batch, int seq, int hidden);

    // Returns per-expert load (fraction of tokens routed to each expert).
    std::vector<float> expert_load() const;
};
