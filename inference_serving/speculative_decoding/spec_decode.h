#pragma once
#include <vector>
// TODO: implement on GPU

// Draft model proposes `draft_len` tokens; verifier accepts/rejects in parallel.
class SpecDecoder {
public:
    SpecDecoder(const std::string& draft_model_path,
                const std::string& verifier_model_path,
                int draft_len = 4);

    // Sample draft_len tokens from draft model
    std::vector<int> propose(const std::vector<int>& prompt_ids);

    // Verify proposed tokens with large model. Returns index of first rejection.
    // If all accepted: returns draft_len (all pass).
    int verify(const std::vector<int>& prompt_ids,
               const std::vector<int>& proposed_tokens);

    double acceptance_rate() const;  // running average
};
