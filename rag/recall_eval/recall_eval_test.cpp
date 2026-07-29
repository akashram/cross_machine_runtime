// recall_eval_test.cpp -- PLAN.md Phase 13 step 6: recall@k against
// rag/corpus/corpus.h's 8 hand-labeled queries, over a 48-document index
// (8 signal documents + 40 distractors -- see corpus.h's own note on why
// the distractors exist: at only 8 documents, ml/knn's BallTree default
// leaf_size makes "approximate" mode identical to exact mode, so there's
// no real retrieval difficulty to measure recall against). Reports
// recall@1/3/5 in exact mode as the primary measurement, plus a same-k
// exact-vs-approximate comparison to set up step 8's deeper study.
#include "recall_eval.h"
#include "../embedding_model/train_encoder.h"

#include <cstdio>
#include <string>

using namespace rag;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

} // namespace

int main() {
  auto docs = sample_documents();
  auto queries = sample_queries();
  auto distractors = sample_distractor_documents();

  std::string extra_vocab;
  for (const auto &d : distractors) extra_vocab += d.text + " ";

  TrainedEncoder encoder = train_corpus_encoder(docs, queries, extra_vocab);
  RagIndex index = build_index(encoder.params, encoder.tokenizer, sample_documents_with_distractors());

  std::printf("  index size: %d chunks (8 signal documents + %d distractors)\n", index.size(),
              static_cast<int>(distractors.size()));

  auto r1 = compute_recall_at_k(index, encoder.params, encoder.tokenizer, queries, 1);
  auto r3 = compute_recall_at_k(index, encoder.params, encoder.tokenizer, queries, 3);
  auto r5 = compute_recall_at_k(index, encoder.params, encoder.tokenizer, queries, 5);

  std::printf("  recall@1 = %.3f (%d/%d)\n", r1.recall, r1.hits, r1.total);
  std::printf("  recall@3 = %.3f (%d/%d)\n", r3.recall, r3.hits, r3.total);
  std::printf("  recall@5 = %.3f (%d/%d)\n", r5.recall, r5.hits, r5.total);

  require(r1.recall >= 0.6, "recall@1 is well above chance (1/48 = 0.021) against a 48-document index");
  require(r3.recall >= r1.recall, "recall@3 is at least as high as recall@1 (larger k can only help)");
  require(r5.recall >= r3.recall, "recall@5 is at least as high as recall@3 (larger k can only help)");

  auto r5_approx = compute_recall_at_k(index, encoder.params, encoder.tokenizer, queries, 5, /*approximate=*/true);
  std::printf("  recall@5 (BallTree approximate/defeatist mode) = %.3f (%d/%d)\n", r5_approx.recall, r5_approx.hits,
              r5_approx.total);
  require(r5_approx.recall <= r5.recall + 1e-9,
          "approximate retrieval's recall@5 never exceeds exact retrieval's (a real ceiling, not just usually true)");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
