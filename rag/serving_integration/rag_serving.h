#pragma once

// PLAN.md Phase 13 step 9: wire the retrieval + generation pipeline into
// inference_serving/serving_backend's ServingRouter (Phase 9 step 8) as a
// new request path. Deliberately does NOT modify serving_router.h/.cpp --
// inference_serving is a Phase 9 component and must stay usable without
// any Phase 13 (rag/) dependency; the dependency direction throughout
// this repo is later phases depending on earlier ones, never the reverse.
// Instead, route_rag() does exactly what rag_generate() (step 5) did,
// EXCEPT the final generation call goes through the caller's EXISTING
// ServingRouter::route() -- CPU/GPU/FPGA/TPU backend selection and
// fallback is entirely ServingRouter's unmodified job; this file only
// adds what happens BEFORE that call (embed the query, retrieve, build
// the augmented prompt).

#include "../../inference_serving/serving_backend/serving_router.h"
#include "../../transformer/transformer_model.h"
#include "../indexing_pipeline/index_pipeline.h"
#include "../rag_generation/prompt_construction.h"

#include <string>
#include <vector>

namespace rag {

struct RagRouteResult {
  inference_serving::Backend backend_used;
  bool fell_back;
  std::string prompt;
  std::vector<std::string> retrieved_chunk_texts;
  std::vector<int> full_tokens; // prompt tokens followed by generated tokens
  std::string continuation_text;
};

inline RagRouteResult route_rag(const inference_serving::ServingRouter &router, inference_serving::Backend preferred,
                                 const RagIndex &index, const EncoderParams &encoder,
                                 const transformer::CharTokenizer &encoder_tok,
                                 const transformer::ModelParams &causal_model,
                                 const transformer::CharTokenizer &causal_tok, const std::string &query, int k,
                                 int max_new_tokens, bool use_retrieval = true) {
  RagRouteResult result;
  if (use_retrieval) {
    for (const auto &r : query_index(index, encoder, encoder_tok, query, k))
      result.retrieved_chunk_texts.push_back(index.chunk(r.index).text);
  }
  result.prompt = construct_prompt(result.retrieved_chunk_texts, query);
  std::vector<int> prompt_tokens = causal_tok.encode(result.prompt);

  inference_serving::RouteResult routed = router.route(preferred, causal_model, prompt_tokens, max_new_tokens);
  result.backend_used = routed.backend_used;
  result.fell_back = routed.fell_back;
  result.full_tokens = routed.tokens;

  std::vector<int> continuation(result.full_tokens.begin() + static_cast<long>(prompt_tokens.size()),
                                 result.full_tokens.end());
  result.continuation_text = causal_tok.decode(continuation);
  return result;
}

} // namespace rag
