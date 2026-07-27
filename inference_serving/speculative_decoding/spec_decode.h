#pragma once
#include "../../transformer/transformer_model.h"

#include <vector>

// PLAN.md Phase 9 step 5: draft model proposes tokens, a larger verifier
// accepts/rejects them in parallel (Chen et al., "Fast Inference from
// Transformers via Speculative Decoding"). Uses transformer/'s real,
// CPU-trained model as both draft and verifier — a smaller/larger config
// of the same architecture, trained on the same corpus — instead of
// stubbing this out or requiring GPU-hosted 1B/7B checkpoints. See
// spec_decode_test.cpp for the training setup and README.md for why this
// diverges from the original stub's `draft_model_path`/
// `verifier_model_path` file-loading constructor.
//
// Acceptance rule: greedy-equivalence. The real algorithm (Chen et al.)
// verifies with rejection SAMPLING against a stochastic draft
// distribution; this repo's draft/verifier are both used greedily
// (deterministic argmax), so a token is "accepted" iff the verifier's own
// argmax at that position matches the draft's proposal — the exact,
// well-defined special case of rejection sampling when both distributions
// are point masses. See README.md's Design section.

namespace inference_serving {

class SpecDecoder {
 public:
  SpecDecoder(transformer::ModelParams draft_model, transformer::ModelParams verifier_model, int draft_len = 4);

  // Greedy-decodes draft_len tokens from the draft model, continuing
  // autoregressively from prompt_ids (fed back into the draft model each
  // step — the draft model, unlike the verifier, IS run token-by-token,
  // since that's the whole point: it's cheap).
  std::vector<int> propose(const std::vector<int> &prompt_ids);

  // ONE verifier forward pass over prompt_ids + proposed_tokens. Returns
  // the count of leading tokens whose verifier-argmax matches the
  // proposal (0..draft_len; draft_len means every proposed token was
  // accepted).
  int verify(const std::vector<int> &prompt_ids, const std::vector<int> &proposed_tokens);

  // One full speculative-decoding round: propose(), verify(), then take
  // the accepted prefix PLUS one bonus token — the verifier's own argmax
  // at the first-rejected (or, if none rejected, the one-past-the-end)
  // position, which that same forward pass already computed for free.
  // Appends the accepted tokens to `context` and returns them.
  std::vector<int> generate_round(std::vector<int> &context);

  double acceptance_rate() const {
    return total_proposed_ == 0 ? 0.0 : static_cast<double>(total_accepted_) / static_cast<double>(total_proposed_);
  }
  long long num_verifier_calls() const { return num_verifier_calls_; }
  long long num_draft_calls() const { return num_draft_calls_; }

 private:
  // Shared by verify() and generate_round(): one verifier forward pass,
  // returning both the accepted-count verify() reports AND the "bonus"
  // token (the verifier's own argmax at the first-rejected position),
  // which generate_round() needs but verify()'s public int-returning
  // signature (matching the original stub) has no room for.
  struct VerifyResult {
    int accepted_count;
    int bonus_token;
  };
  VerifyResult verify_full(const std::vector<int> &prompt_ids, const std::vector<int> &proposed_tokens);

  transformer::ModelParams draft_;
  transformer::ModelParams verifier_;
  int draft_len_;
  long long total_proposed_ = 0;
  long long total_accepted_ = 0;
  long long num_verifier_calls_ = 0;
  long long num_draft_calls_ = 0;
};

}  // namespace inference_serving
