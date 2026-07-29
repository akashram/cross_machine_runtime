#pragma once

// PLAN.md Phase 13 step 5 (generation half): embed the query, retrieve
// top-k chunks (indexing_pipeline, step 4), construct the augmented
// context window (prompt_construction.h), and feed it into the EXISTING
// causal generation path -- inference_serving's make_cpu_backend()
// (Phase 9 step 8), not a third reimplementation of the greedy-decode
// loop transformer_test.cpp and serving_router.cpp already have.
//
// Scope: this step is the WIRING (retrieval -> prompt -> generation
// composes correctly), verified with whatever encoder/causal model the
// caller passes in -- it does not itself judge generation QUALITY (that's
// step 7's job, with a causal model actually trained for the
// answer-from-context skill).

#include "prompt_construction.h"
#include "../../inference_serving/serving_backend/serving_router.h"
#include "../../transformer/transformer_model.h"
#include "../indexing_pipeline/index_pipeline.h"

#include <string>
#include <vector>

namespace rag {

struct RagGenerationResult {
  std::string prompt;
  std::vector<std::string> retrieved_chunk_texts;
  std::vector<int> full_tokens; // prompt tokens followed by generated tokens
  std::string continuation_text; // decoded text generated AFTER the prompt
};

inline RagGenerationResult rag_generate(const RagIndex &index, const EncoderParams &encoder,
                                         const transformer::CharTokenizer &encoder_tok,
                                         const transformer::ModelParams &causal_model,
                                         const transformer::CharTokenizer &causal_tok, const std::string &query, int k,
                                         int max_new_tokens, bool use_retrieval = true) {
  RagGenerationResult result;
  if (use_retrieval) {
    for (const auto &r : query_index(index, encoder, encoder_tok, query, k))
      result.retrieved_chunk_texts.push_back(index.chunk(r.index).text);
  }
  result.prompt = construct_prompt(result.retrieved_chunk_texts, query);

  std::vector<int> prompt_tokens = causal_tok.encode(result.prompt);
  static const auto generate_fn = inference_serving::make_cpu_backend();
  result.full_tokens = generate_fn(causal_model, prompt_tokens, max_new_tokens);

  std::vector<int> continuation(result.full_tokens.begin() + static_cast<long>(prompt_tokens.size()),
                                 result.full_tokens.end());
  result.continuation_text = causal_tok.decode(continuation);
  return result;
}

} // namespace rag
