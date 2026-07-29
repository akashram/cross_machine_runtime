#pragma once

// PLAN.md Phase 13 step 5 (prompt-construction half): given a query and a
// list of already-retrieved chunk texts, build the augmented context
// window fed to the causal generation path. Deliberately just string
// assembly -- the retrieval decision (which chunks, how many) already
// happened in indexing_pipeline's query_index(); this is the template.

#include <string>
#include <vector>

namespace rag {

inline std::string construct_prompt(const std::vector<std::string> &retrieved_chunks, const std::string &query) {
  std::string prompt = "Context:";
  for (const auto &chunk : retrieved_chunks) {
    prompt += " ";
    prompt += chunk;
  }
  prompt += " Question: ";
  prompt += query;
  prompt += " Answer:";
  return prompt;
}

} // namespace rag
