#pragma once

// Trains a Siamese encoder (step 1's mechanism) on rag/corpus/corpus.h's
// shared 8-topic (query, document) pairs -- reused by steps 4, 6, 7, and 8
// so every one of them retrieves against the identical trained index
// instead of each training its own encoder on its own ad hoc data.
// Depends on RagEmbedding, RagCorpus, and Transformer (for CharTokenizer);
// link all three.

#include "embedding_model.h"
#include "../../transformer/char_tokenizer.h"
#include "../corpus/corpus.h"

#include <random>
#include <string>

namespace rag {

struct TrainedEncoder {
  EncoderParams params;
  transformer::CharTokenizer tokenizer;
};

// Hyperparameters tuned for wall-clock, not just accuracy: attention cost
// is O(seq^2 * d_model), and this corpus's documents (~70-90 characters)
// are ~4x longer than embedding_model_test's toy 6-pair corpus -- ~15x
// more attention compute per layer from sequence length alone. A single
// layer and a smaller d_model/d_ff keep a full training run within a few
// seconds instead of several minutes, while still reaching high in-batch
// retrieval accuracy (see indexing_pipeline/README.md's measured numbers).
// `extra_vocab_text`: appended to the tokenizer's vocab-building corpus
// ONLY -- not used as training data -- so callers that need to encode
// additional text through the same tokenizer later (e.g. recall_eval's
// distractor documents, bulking up the index without being part of the
// contrastive training signal) don't hit an "unseen character" error.
inline TrainedEncoder train_corpus_encoder(const std::vector<Document> &docs, const std::vector<QueryJudgment> &queries,
                                            const std::string &extra_vocab_text = "", int d_model = 16,
                                            int num_heads = 2, int num_layers = 1, int d_ff = 32, int embed_dim = 10,
                                            int epochs = 200, float lr = 0.15f, float temperature = 0.15f,
                                            unsigned init_seed = 21) {
  std::string joint;
  for (const auto &d : docs) joint += d.text + " ";
  for (const auto &q : queries) joint += q.query + " ";
  joint += extra_vocab_text;
  transformer::CharTokenizer tok(joint);

  EncoderConfig cfg{tok.vocab_size(), d_model, num_heads, num_layers, d_ff, /*max_seq_len=*/128, embed_dim};
  std::mt19937 rng(init_seed);
  EncoderParams params = init_encoder(cfg, rng);

  int batch = static_cast<int>(queries.size());
  std::vector<std::vector<int>> query_tok, doc_tok;
  for (const auto &q : queries) {
    query_tok.push_back(tok.encode(q.query));
    doc_tok.push_back(tok.encode(docs[static_cast<std::size_t>(q.relevant_doc_id)].text));
  }

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::vector<EncoderCache> q_caches(static_cast<std::size_t>(batch)), d_caches(static_cast<std::size_t>(batch));
    std::vector<Matrix> q_embs, d_embs;
    for (int i = 0; i < batch; ++i)
      q_embs.push_back(encode_forward(params, query_tok[static_cast<std::size_t>(i)], q_caches[static_cast<std::size_t>(i)]));
    for (int i = 0; i < batch; ++i)
      d_embs.push_back(encode_forward(params, doc_tok[static_cast<std::size_t>(i)], d_caches[static_cast<std::size_t>(i)]));

    auto nce = info_nce_loss(q_embs, d_embs, temperature);

    EncoderGrads grad = zero_encoder_grad(cfg);
    for (int i = 0; i < batch; ++i) {
      EncoderGrads gi = zero_encoder_grad(cfg);
      encode_backward(params, q_caches[static_cast<std::size_t>(i)], nce.d_query_emb[static_cast<std::size_t>(i)], gi);
      accumulate_encoder_grad(grad, gi);
      EncoderGrads gj = zero_encoder_grad(cfg);
      encode_backward(params, d_caches[static_cast<std::size_t>(i)], nce.d_doc_emb[static_cast<std::size_t>(i)], gj);
      accumulate_encoder_grad(grad, gj);
    }
    encoder_sgd_step(params, grad, lr);
  }

  return TrainedEncoder{params, tok};
}

} // namespace rag
