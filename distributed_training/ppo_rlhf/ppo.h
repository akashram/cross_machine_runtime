#pragma once
// TODO: implement on GPU (requires SFT model + reward model)

struct RlhfBatch {
    const int*   prompt_ids;     // [batch, prompt_len]
    const float* response_logprobs;  // [batch, response_len] from policy
    const float* ref_logprobs;       // [batch, response_len] from frozen ref model
    const float* rewards;            // [batch] from reward model
    int batch_size, prompt_len, response_len;
};

class PPOTrainer {
public:
    PPOTrainer(const std::string& policy_path,   // SFT init
               const std::string& ref_path,       // frozen reference
               float kl_coef = 0.05f,
               float clip_ratio = 0.2f);

    // One PPO step. Returns: policy loss, value loss, KL divergence, mean reward.
    struct PPOStats { float policy_loss, value_loss, kl, mean_reward; };
    PPOStats train_step(const RlhfBatch& batch);
};
