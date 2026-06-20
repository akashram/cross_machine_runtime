#pragma once
#include <string>
// TODO: implement on GPU

struct SftBatch {
    const int*   input_ids;    // [batch, seq_len]
    const int*   labels;       // [batch, seq_len] — -100 for prompt tokens (masked)
    const float* attention_mask;
    int          batch_size, seq_len;
};

class SftTrainer {
public:
    SftTrainer(const std::string& model_path, int rank, int world_size);

    // One gradient step. Returns cross-entropy loss on response tokens only.
    float train_step(const SftBatch& batch);

    void save_checkpoint(const std::string& path);
};
