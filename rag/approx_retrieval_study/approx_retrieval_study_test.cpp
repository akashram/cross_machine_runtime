// approx_retrieval_study_test.cpp -- PLAN.md Phase 13 step 8: does
// ml/knn's already-established approximate/exact recall-speed tradeoff
// (BallTree's defeatist mode -- see ml/knn/README.md, and recall_eval's
// own 1.000 -> 0.875 recall@5 drop on this exact corpus) actually degrade
// end-task GENERATION quality, or is the imperfect recall "free" at the
// system level once a real causal model is reading the (possibly wrong)
// retrieved context?
//
// Pure composition: reuses recall_eval's compute_recall_at_k and
// generation_quality's measure_qa_accuracy, both of which already accept
// an `approximate` flag -- no new retrieval or generation logic, only the
// comparison itself is new.
#include "../generation_quality/causal_qa_model.h"
#include "../embedding_model/train_encoder.h"
#include "../recall_eval/recall_eval.h"

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

  // Retrieval-only view: recall@1 (the k actually used to pick generation
  // context below), exact vs. BallTree's approximate/defeatist mode.
  auto recall_exact = compute_recall_at_k(index, encoder.params, encoder.tokenizer, queries, /*k=*/1, /*approximate=*/false);
  auto recall_approx = compute_recall_at_k(index, encoder.params, encoder.tokenizer, queries, /*k=*/1, /*approximate=*/true);
  std::printf("  recall@1 exact:       %.3f (%d/%d)\n", recall_exact.recall, recall_exact.hits, recall_exact.total);
  std::printf("  recall@1 approximate: %.3f (%d/%d)\n", recall_approx.recall, recall_approx.hits, recall_approx.total);

  // System-level view: does the causal model actually generating the
  // correct answer degrade by the SAME amount, a SMALLER amount (wrong
  // context still accidentally helpful, or the model falling back to
  // reasonable behavior), or a LARGER amount (compounding)?
  auto gen_exact = measure_qa_accuracy(index, encoder.params, encoder.tokenizer, causal, queries, /*k=*/1, /*approximate=*/false);
  auto gen_approx = measure_qa_accuracy(index, encoder.params, encoder.tokenizer, causal, queries, /*k=*/1, /*approximate=*/true);
  std::printf("  generation accuracy exact:       %.3f\n", gen_exact.with_retrieval_accuracy);
  std::printf("  generation accuracy approximate: %.3f\n", gen_approx.with_retrieval_accuracy);

  double recall_drop = recall_exact.recall - recall_approx.recall;
  double generation_drop = gen_exact.with_retrieval_accuracy - gen_approx.with_retrieval_accuracy;
  std::printf("  recall@1 drop from approximation:            %.3f\n", recall_drop);
  std::printf("  generation accuracy drop from approximation: %.3f\n", generation_drop);

  require(gen_exact.with_retrieval_accuracy >= gen_approx.with_retrieval_accuracy,
          "approximate retrieval's end-task generation accuracy never exceeds exact retrieval's");
  require(recall_approx.recall <= recall_exact.recall, "approximate retrieval's recall@1 never exceeds exact retrieval's");

  // The real, honest question this step exists to answer: is a recall
  // drop "free" at the system level, or does it cost end-task accuracy?
  // Deliberately NOT asserted in a fixed direction -- that IS the
  // question, not an assumed answer. Both a report finding "free" (recall
  // drop absorbed, generation still correct) and one finding "not free"
  // (recall drop directly costs generation accuracy) are real, valid
  // outcomes; only the printed numbers and README's interpretation of
  // THIS run's actual measurement decide which one it was.
  if (generation_drop < recall_drop - 1e-9)
    std::printf("  finding: this recall drop was PARTIALLY absorbed -- generation accuracy fell less than recall did\n");
  else if (generation_drop > recall_drop + 1e-9)
    std::printf("  finding: this recall drop COMPOUNDED -- generation accuracy fell more than recall did\n");
  else
    std::printf("  finding: generation accuracy fell by exactly the recall drop -- not free, not compounded\n");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
