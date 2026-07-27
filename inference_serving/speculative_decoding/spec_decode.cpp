#include "spec_decode.h"

namespace inference_serving {

namespace {
int argmax_at(const transformer::Matrix &logits, int row, int vocab_size) {
  int best_idx = 0;
  float best_val = logits(row, 0);
  for (int v = 1; v < vocab_size; ++v) {
    if (logits(row, v) > best_val) {
      best_val = logits(row, v);
      best_idx = v;
    }
  }
  return best_idx;
}
}  // namespace

SpecDecoder::SpecDecoder(transformer::ModelParams draft_model, transformer::ModelParams verifier_model, int draft_len)
    : draft_(std::move(draft_model)), verifier_(std::move(verifier_model)), draft_len_(draft_len) {}

std::vector<int> SpecDecoder::propose(const std::vector<int> &prompt_ids) {
  std::vector<int> context = prompt_ids;
  std::vector<int> proposed;
  proposed.reserve(static_cast<std::size_t>(draft_len_));
  for (int i = 0; i < draft_len_; ++i) {
    transformer::ModelCache cache;
    transformer::Matrix logits = transformer::model_forward(draft_, context, cache);
    ++num_draft_calls_;
    int last = static_cast<int>(context.size()) - 1;
    int token = argmax_at(logits, last, draft_.config.vocab_size);
    proposed.push_back(token);
    context.push_back(token);
  }
  return proposed;
}

SpecDecoder::VerifyResult SpecDecoder::verify_full(const std::vector<int> &prompt_ids,
                                                     const std::vector<int> &proposed_tokens) {
  std::vector<int> full_seq = prompt_ids;
  full_seq.insert(full_seq.end(), proposed_tokens.begin(), proposed_tokens.end());

  transformer::ModelCache cache;
  transformer::Matrix logits = transformer::model_forward(verifier_, full_seq, cache);
  ++num_verifier_calls_;

  int context_len = static_cast<int>(prompt_ids.size());
  int accepted = 0;
  int bonus_token = 0;
  for (int i = 0; i < static_cast<int>(proposed_tokens.size()); ++i) {
    int pos = context_len + i - 1;
    int verifier_argmax = argmax_at(logits, pos, verifier_.config.vocab_size);
    if (verifier_argmax == proposed_tokens[static_cast<std::size_t>(i)]) {
      ++accepted;
    } else {
      bonus_token = verifier_argmax;  // the free correction at the rejection point
      break;
    }
  }
  if (accepted == static_cast<int>(proposed_tokens.size())) {
    // Every proposed token matched: the bonus token is the verifier's
    // prediction one past the last proposed token -- also already
    // computed by this same forward pass.
    int pos = context_len + accepted - 1;
    bonus_token = argmax_at(logits, pos, verifier_.config.vocab_size);
  }

  total_proposed_ += static_cast<long long>(proposed_tokens.size());
  total_accepted_ += accepted;
  return {accepted, bonus_token};
}

int SpecDecoder::verify(const std::vector<int> &prompt_ids, const std::vector<int> &proposed_tokens) {
  return verify_full(prompt_ids, proposed_tokens).accepted_count;
}

std::vector<int> SpecDecoder::generate_round(std::vector<int> &context) {
  std::vector<int> proposed = propose(context);
  VerifyResult result = verify_full(context, proposed);

  std::vector<int> accepted_tokens(proposed.begin(), proposed.begin() + result.accepted_count);
  accepted_tokens.push_back(result.bonus_token);

  context.insert(context.end(), accepted_tokens.begin(), accepted_tokens.end());
  return accepted_tokens;
}

}  // namespace inference_serving
