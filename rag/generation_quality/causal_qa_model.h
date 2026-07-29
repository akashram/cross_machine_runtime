#pragma once

// PLAN.md Phase 13 step 7: a small causal QA model trained on
// rag/corpus/corpus.h's queries in two conditions per query --
// ("q: <query> c: <its true document text> a: <answer>.") and
// ("q: <query> c: none a: unknown.") -- so it learns to condition its
// answer on whatever context text actually appears in the prompt, and to
// abstain when there isn't any. Reused by generation_quality_test.cpp
// (step 7) and approx_retrieval_study (step 8), which both need the
// identical trained model to compare fairly.
//
// Real, disclosed scope limitation: at this toy scale (8 queries, tiny
// char-level model), the mechanism here is closer to memorization of 16
// (prompt -> continuation) mappings than a generalizable "read the
// context and answer" skill that would transfer to an unseen query. What
// IS real and load-bearing: the model's output is causally driven by
// which TEXT is actually present in the prompt (context vs "none"), not
// hard-coded per query -- so feeding it the correct RETRIEVED context
// (via the real embedding + ANN pipeline, not injected ground truth)
// still requires retrieval to have actually worked.

#include "../../inference_serving/serving_backend/serving_router.h"
#include "../../transformer/char_tokenizer.h"
#include "../../transformer/transformer_model.h"
#include "../corpus/corpus.h"
#include "../indexing_pipeline/index_pipeline.h"

#include <random>
#include <string>
#include <vector>

namespace rag {

struct TrainedCausalModel {
  transformer::ModelParams params;
  transformer::CharTokenizer tokenizer;
};

inline std::string construct_qa_prompt(const std::string &query, const std::string &context_or_none) {
  return "q: " + query + " c: " + context_or_none + " a:";
}

inline TrainedCausalModel train_qa_model(const std::vector<Document> &docs, const std::vector<QueryJudgment> &queries,
                                          const std::string &extra_vocab_text = "", int d_model = 16,
                                          int num_heads = 2, int num_layers = 1, int d_ff = 32, int epochs = 150,
                                          float lr = 0.15f, unsigned seed = 17) {
  std::vector<std::string> texts;
  for (const auto &q : queries) {
    const std::string &context = docs[static_cast<std::size_t>(q.relevant_doc_id)].text;
    texts.push_back(construct_qa_prompt(q.query, context) + " " + q.answer + ".");
    texts.push_back(construct_qa_prompt(q.query, "none") + " unknown.");
  }
  std::string joint;
  for (const auto &t : texts) joint += t + " ";
  joint += extra_vocab_text;
  transformer::CharTokenizer tok(joint);

  transformer::TransformerConfig cfg{tok.vocab_size(), d_model, num_heads, num_layers, d_ff, /*max_seq_len=*/200};
  std::mt19937 rng(seed);
  transformer::ModelParams model = transformer::init_model(cfg, rng);

  std::vector<std::vector<int>> token_seqs;
  for (const auto &t : texts) token_seqs.push_back(tok.encode(t));

  for (int epoch = 0; epoch < epochs; ++epoch) {
    for (const auto &tokens : token_seqs) {
      transformer::ModelCache cache;
      transformer::Matrix logits = transformer::model_forward(model, tokens, cache);
      auto loss = transformer::next_token_loss(logits, tokens);
      transformer::ModelGrads grad = transformer::zero_model_grad(cfg);
      transformer::model_backward(model, cache, loss.dlogits, grad);
      transformer::sgd_step(model, grad, lr);
    }
  }
  return TrainedCausalModel{model, tok};
}

inline bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

inline std::string generate_answer(const TrainedCausalModel &causal, const std::string &prompt, int max_new_tokens = 15) {
  std::vector<int> prompt_tokens = causal.tokenizer.encode(prompt);
  static const auto generate_fn = inference_serving::make_cpu_backend();
  std::vector<int> full = generate_fn(causal.params, prompt_tokens, max_new_tokens);
  std::vector<int> continuation(full.begin() + static_cast<long>(prompt_tokens.size()), full.end());
  return causal.tokenizer.decode(continuation);
}

// Teacher-forced next-token loss on a FULL constructed sequence (not
// restricted to just the answer span) -- a simple, honestly-scoped
// perplexity-style signal: how surprised the model is by an entire
// "q: ... c: ... a: <answer>." sequence, with vs. without real context.
inline float sequence_loss(const TrainedCausalModel &causal, const std::string &text) {
  std::vector<int> tokens = causal.tokenizer.encode(text);
  transformer::ModelCache cache;
  transformer::Matrix logits = transformer::model_forward(causal.params, tokens, cache);
  return transformer::next_token_loss(logits, tokens).loss;
}

struct QAAccuracyResult {
  double with_retrieval_accuracy;
  double without_retrieval_accuracy;
};

// Generation-based accuracy: does the model's greedy continuation after
// "a:" contain the query's true answer substring? Measured once WITH
// real retrieval (embed query -> query_index -> top-1 chunk text as
// context) and once with context forced to "none" -- the with-vs-without
// ablation PLAN.md step 7 asks for.
inline QAAccuracyResult measure_qa_accuracy(const RagIndex &index, const EncoderParams &encoder,
                                             const transformer::CharTokenizer &encoder_tok,
                                             const TrainedCausalModel &causal, const std::vector<QueryJudgment> &queries,
                                             int k = 1, bool approximate = false) {
  int with_correct = 0, without_correct = 0;
  for (const auto &q : queries) {
    auto results = query_index(index, encoder, encoder_tok, q.query, k, approximate);
    std::string context = results.empty() ? "none" : index.chunk(results[0].index).text;

    std::string with_continuation = generate_answer(causal, construct_qa_prompt(q.query, context));
    if (contains(with_continuation, q.answer)) ++with_correct;

    std::string without_continuation = generate_answer(causal, construct_qa_prompt(q.query, "none"));
    if (contains(without_continuation, q.answer)) ++without_correct;
  }
  int n = static_cast<int>(queries.size());
  return {static_cast<double>(with_correct) / n, static_cast<double>(without_correct) / n};
}

} // namespace rag
