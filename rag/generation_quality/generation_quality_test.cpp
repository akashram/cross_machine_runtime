// generation_quality_test.cpp -- PLAN.md Phase 13 step 7: does retrieval
// measurably help the causal model answer questions it can't answer from
// training data alone? Two complementary measurements:
//  1. generation-based accuracy: does greedy-decoded continuation after
//     "a:" contain the true answer, WITH real retrieval (embed query,
//     search the 48-document index, use the actual top-1 chunk as
//     context) vs. WITHOUT (context forced to "none")?
//  2. a simple perplexity-style signal: teacher-forced sequence loss on
//     "q: ... c: ... a: <TRUE answer>." with real context vs. with
//     context forced to "none" -- the without-context version was NEVER
//     seen in training (training's no-context examples all end in
//     "unknown.", not the true answer), so this is a real held-out
//     likelihood probe, not just re-scoring a memorized string.
#include "causal_qa_model.h"
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
  TrainedCausalModel causal = train_qa_model(docs, queries);

  auto acc = measure_qa_accuracy(index, encoder.params, encoder.tokenizer, causal, queries);
  std::printf("  generation accuracy WITH retrieval:    %.3f\n", acc.with_retrieval_accuracy);
  std::printf("  generation accuracy WITHOUT retrieval: %.3f\n", acc.without_retrieval_accuracy);

  require(acc.with_retrieval_accuracy >= 0.75, "with real retrieved context, the model generates the correct answer for most queries");
  require(acc.without_retrieval_accuracy <= 0.25, "without any context, the model does NOT fabricate the correct answer for most queries");
  require(acc.with_retrieval_accuracy > acc.without_retrieval_accuracy,
          "retrieval measurably improves generation accuracy on this query set");

  double sum_loss_with = 0.0, sum_loss_without = 0.0;
  for (const auto &q : queries) {
    const std::string &true_context = docs[static_cast<std::size_t>(q.relevant_doc_id)].text;
    sum_loss_with += static_cast<double>(sequence_loss(causal, construct_qa_prompt(q.query, true_context) + " " + q.answer + "."));
    // Held-out: the model was trained on "c: none a: unknown.", never
    // "c: none a: <true answer>." -- forcing the true answer here probes
    // genuine generalization, not a memorized string.
    sum_loss_without += static_cast<double>(sequence_loss(causal, construct_qa_prompt(q.query, "none") + " " + q.answer + "."));
  }
  double avg_loss_with = sum_loss_with / static_cast<double>(queries.size());
  double avg_loss_without = sum_loss_without / static_cast<double>(queries.size());
  std::printf("  avg sequence loss forcing the TRUE answer, WITH context:    %.4f\n", avg_loss_with);
  std::printf("  avg sequence loss forcing the TRUE answer, WITHOUT context: %.4f\n", avg_loss_without);
  require(avg_loss_without > avg_loss_with,
          "the model is more 'surprised' by the correct answer when no context is given (a real perplexity-style retrieval benefit)");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
