#pragma once

// PLAN.md Phase 13 step 6: recall@k measured against a real
// relevance-labeled query set (rag/corpus/corpus.h's QueryJudgments) --
// not just "the index runs." Reused by step 8, which needs the identical
// measurement under approximate retrieval to isolate retrieval's effect
// on end-task generation quality.

#include "../corpus/corpus.h"
#include "../indexing_pipeline/index_pipeline.h"

#include <cstddef>

namespace rag {

struct RecallResult {
  double recall;
  int hits;
  int total;
};

// Whether query q's true relevant document appears ANYWHERE in its top-k
// retrieved results (this corpus: exactly one relevant document per
// query, so "hit" is a single boolean per query, not a graded score).
inline bool is_hit(const RagIndex &index, const std::vector<CosineNeighborResult> &results, int relevant_doc_id) {
  for (const auto &r : results)
    if (index.chunk(r.index).doc_id == relevant_doc_id) return true;
  return false;
}

inline RecallResult compute_recall_at_k(const RagIndex &index, const EncoderParams &encoder,
                                         const transformer::CharTokenizer &tok, const std::vector<QueryJudgment> &queries,
                                         int k, bool approximate = false) {
  int hits = 0;
  for (const auto &q : queries) {
    auto results = query_index(index, encoder, tok, q.query, k, approximate);
    if (is_hit(index, results, q.relevant_doc_id)) ++hits;
  }
  int total = static_cast<int>(queries.size());
  return RecallResult{static_cast<double>(hits) / total, hits, total};
}

} // namespace rag
