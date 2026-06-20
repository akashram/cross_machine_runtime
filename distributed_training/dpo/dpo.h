#pragma once
// TODO: implement on GPU (alternative to PPO — no reward model needed)

struct PreferenceBatch {
    const int*   chosen_ids;    // [batch, seq_len] — preferred response
    const int*   rejected_ids;  // [batch, seq_len] — dispreferred response
    int batch_size, seq_len;
};

class DPOTrainer {
public:
    DPOTrainer(const std::string& policy_path,  // SFT init
               const std::string& ref_path,      // frozen reference
               float beta = 0.1f);               // KL regularization strength

    // DPO gradient step. Returns: loss, chosen_reward, rejected_reward.
    struct DPOStats { float loss, chosen_reward, rejected_reward; };
    DPOStats train_step(const PreferenceBatch& batch);
};
