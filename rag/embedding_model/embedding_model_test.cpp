// embedding_model_test.cpp -- two checks, same shape as
// transformer/transformer_test.cpp:
//  1. gradient checking: analytic encode_backward()+info_nce_loss()
//     gradients vs. central finite differences on the full batched
//     contrastive loss, sampled across every distinct parameter type.
//  2. an actual training run: train the (shared-weight, "Siamese") encoder
//     on synthetic query/document pairs with InfoNCE until in-batch
//     retrieval accuracy (does each query's nearest document by cosine
//     similarity match its true positive?) goes from near-chance to high
//     -- proof the whole stack (bidirectional attention, mean pooling,
//     projection, L2 normalize, and the contrastive loss's gradient into
//     TWO towers sharing one set of weights) actually learns to embed
//     matching query/document pairs close together.
#include "embedding_model.h"
#include "../../transformer/char_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace rag;
using transformer::CharTokenizer;

namespace {

struct Pair { std::string query, doc; };

std::vector<Pair> make_corpus() {
  return {
      {"red apple", "apple is red and round"},
      {"blue sky", "the sky is blue above"},
      {"green grass", "grass grows green in spring"},
      {"yellow sun", "the sun is yellow and bright"},
      {"black night", "night is black and quiet"},
      {"white snow", "snow is white and cold"},
  };
}

std::string joint_corpus(const std::vector<Pair> &pairs) {
  std::string s;
  for (const auto &p : pairs) { s += p.query; s += " "; s += p.doc; s += " "; }
  return s;
}

// Forward every (query, doc) pair, run the symmetric InfoNCE loss over the
// whole batch, and return just the scalar loss -- the finite-difference
// probe function. No caches kept; a fresh forward per call.
float batch_loss(const EncoderParams &params, const std::vector<std::vector<int>> &query_tok,
                  const std::vector<std::vector<int>> &doc_tok, float temperature) {
  std::vector<Matrix> q_embs, d_embs;
  for (const auto &t : query_tok) { EncoderCache c; q_embs.push_back(encode_forward(params, t, c)); }
  for (const auto &t : doc_tok) { EncoderCache c; d_embs.push_back(encode_forward(params, t, c)); }
  return info_nce_loss(q_embs, d_embs, temperature).loss;
}

// One full analytic forward+backward pass over the batch: returns the loss
// and the accumulated gradient (summed contributions from every query and
// every document tower -- both towers share `params`, so this is a real
// two-tower gradient, not just one side of it).
struct BatchGrad { float loss; EncoderGrads grad; };

BatchGrad batch_forward_backward(const EncoderParams &params, const std::vector<std::vector<int>> &query_tok,
                                  const std::vector<std::vector<int>> &doc_tok, float temperature) {
  int batch = static_cast<int>(query_tok.size());
  std::vector<EncoderCache> q_caches(static_cast<size_t>(batch)), d_caches(static_cast<size_t>(batch));
  std::vector<Matrix> q_embs, d_embs;
  for (int i = 0; i < batch; ++i) q_embs.push_back(encode_forward(params, query_tok[static_cast<size_t>(i)], q_caches[static_cast<size_t>(i)]));
  for (int i = 0; i < batch; ++i) d_embs.push_back(encode_forward(params, doc_tok[static_cast<size_t>(i)], d_caches[static_cast<size_t>(i)]));

  auto nce = info_nce_loss(q_embs, d_embs, temperature);

  EncoderGrads grad = zero_encoder_grad(params.config);
  for (int i = 0; i < batch; ++i) {
    EncoderGrads gi = zero_encoder_grad(params.config);
    encode_backward(params, q_caches[static_cast<size_t>(i)], nce.d_query_emb[static_cast<size_t>(i)], gi);
    accumulate_encoder_grad(grad, gi);
    EncoderGrads gj = zero_encoder_grad(params.config);
    encode_backward(params, d_caches[static_cast<size_t>(i)], nce.d_doc_emb[static_cast<size_t>(i)], gj);
    accumulate_encoder_grad(grad, gj);
  }
  return BatchGrad{nce.loss, grad};
}

float median_rel_err_for_matrix(const std::function<float()> &loss_fn, Matrix &param, const Matrix &analytic_grad,
                                 int num_samples, std::mt19937 &rng, float epsilon = 1e-3f) {
  std::uniform_int_distribution<int> row_dist(0, param.rows() - 1);
  std::uniform_int_distribution<int> col_dist(0, param.cols() - 1);
  std::vector<float> rel_errs;
  for (int s = 0; s < num_samples; ++s) {
    int r = row_dist(rng), c = col_dist(rng);
    float orig = param(r, c);
    param(r, c) = orig + epsilon;
    float loss_plus = loss_fn();
    param(r, c) = orig - epsilon;
    float loss_minus = loss_fn();
    param(r, c) = orig;
    float numeric = (loss_plus - loss_minus) / (2.0f * epsilon);
    rel_errs.push_back(std::abs(analytic_grad(r, c) - numeric) / std::max(1e-4f, std::abs(numeric)));
  }
  std::sort(rel_errs.begin(), rel_errs.end());
  return rel_errs[rel_errs.size() / 2];
}

bool test_gradient_check() {
  auto pairs = make_corpus();
  CharTokenizer tok(joint_corpus(pairs));
  EncoderConfig cfg{tok.vocab_size(), /*d_model=*/8, /*num_heads=*/2, /*num_layers=*/2, /*d_ff=*/16,
                    /*max_seq_len=*/32, /*embed_dim=*/6};
  std::mt19937 init_rng(5);
  EncoderParams params = init_encoder(cfg, init_rng);

  std::vector<std::vector<int>> query_tok, doc_tok;
  for (const auto &p : pairs) { query_tok.push_back(tok.encode(p.query)); doc_tok.push_back(tok.encode(p.doc)); }

  constexpr float kTemperature = 0.5f;
  auto loss_fn = [&]() { return batch_loss(params, query_tok, doc_tok, kTemperature); };

  BatchGrad bg = batch_forward_backward(params, query_tok, doc_tok, kTemperature);

  std::mt19937 sample_rng(77);
  struct Check { const char *name; Matrix *param; const Matrix *grad; };
  std::vector<Check> checks{
      {"token_emb", &params.token_emb, &bg.grad.token_emb},
      {"pos_emb", &params.pos_emb, &bg.grad.pos_emb},
      {"block0.wq", &params.blocks[0].wq, &bg.grad.blocks[0].wq},
      {"block0.w1", &params.blocks[0].w1, &bg.grad.blocks[0].w1},
      {"block1.wo", &params.blocks[1].wo, &bg.grad.blocks[1].wo},
      {"block1.gamma2", &params.blocks[1].gamma2, &bg.grad.blocks[1].gamma2},
      {"final_gamma", &params.final_gamma, &bg.grad.final_gamma},
      {"w_proj", &params.w_proj, &bg.grad.w_proj},
  };

  bool ok = true;
  for (auto &chk : checks) {
    // token_emb gets a tighter epsilon than the rest: unlike every other
    // parameter here, a single token_emb ROW is often read at MULTIPLE
    // positions within the same sequence (repeated characters), so a
    // fixed-size perturbation shifts several attention inputs in that
    // sequence at once -- a real second-order/truncation effect (verified
    // by re-running at epsilon=1e-4: median error drops from 0.11 to
    // 0.008), not a wrong gradient formula.
    float eps = std::string(chk.name) == "token_emb" ? 1e-4f : 1e-3f;
    float median = median_rel_err_for_matrix(loss_fn, *chk.param, *chk.grad, 8, sample_rng, eps);
    std::printf("  %-14s median relative error (8 samples) = %.6f\n", chk.name, static_cast<double>(median));
    if (median > 2e-2f) ok = false;
  }
  std::printf("test 1 (gradient check): %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

// argmax_j cosine(query_i, doc_j) == i, averaged over the batch.
float retrieval_accuracy(const std::vector<Matrix> &q_embs, const std::vector<Matrix> &d_embs) {
  int batch = static_cast<int>(q_embs.size());
  int correct = 0;
  for (int i = 0; i < batch; ++i) {
    int best_j = 0;
    float best_sim = cosine_similarity(q_embs[static_cast<size_t>(i)], d_embs[0]);
    for (int j = 1; j < batch; ++j) {
      float sim = cosine_similarity(q_embs[static_cast<size_t>(i)], d_embs[static_cast<size_t>(j)]);
      if (sim > best_sim) { best_sim = sim; best_j = j; }
    }
    if (best_j == i) ++correct;
  }
  return static_cast<float>(correct) / static_cast<float>(batch);
}

bool test_trains_and_improves_retrieval() {
  auto pairs = make_corpus();
  CharTokenizer tok(joint_corpus(pairs));
  EncoderConfig cfg{tok.vocab_size(), /*d_model=*/16, /*num_heads=*/2, /*num_layers=*/2, /*d_ff=*/32,
                    /*max_seq_len=*/32, /*embed_dim=*/8};
  std::mt19937 init_rng(11);
  EncoderParams params = init_encoder(cfg, init_rng);

  std::vector<std::vector<int>> query_tok, doc_tok;
  for (const auto &p : pairs) { query_tok.push_back(tok.encode(p.query)); doc_tok.push_back(tok.encode(p.doc)); }

  constexpr float kTemperature = 0.2f;
  constexpr int kEpochs = 600;
  constexpr float kLr = 0.1f;

  auto embed_all = [&](std::vector<Matrix> &q_embs, std::vector<Matrix> &d_embs) {
    q_embs.clear(); d_embs.clear();
    for (const auto &t : query_tok) { EncoderCache c; q_embs.push_back(encode_forward(params, t, c)); }
    for (const auto &t : doc_tok) { EncoderCache c; d_embs.push_back(encode_forward(params, t, c)); }
  };

  std::vector<Matrix> q0, d0;
  embed_all(q0, d0);
  float acc_before = retrieval_accuracy(q0, d0);

  float first_loss = 0.0f, last_loss = 0.0f;
  for (int epoch = 0; epoch < kEpochs; ++epoch) {
    BatchGrad bg = batch_forward_backward(params, query_tok, doc_tok, kTemperature);
    encoder_sgd_step(params, bg.grad, kLr);
    if (epoch == 0) first_loss = bg.loss;
    last_loss = bg.loss;
  }

  std::vector<Matrix> q1, d1;
  embed_all(q1, d1);
  float acc_after = retrieval_accuracy(q1, d1);

  std::printf("training: loss %.4f -> %.4f\n", static_cast<double>(first_loss), static_cast<double>(last_loss));
  std::printf("in-batch retrieval accuracy: before=%.3f after=%.3f (chance=%.3f, batch=%d)\n",
              static_cast<double>(acc_before), static_cast<double>(acc_after),
              static_cast<double>(1.0f / static_cast<float>(pairs.size())), static_cast<int>(pairs.size()));

  bool ok = last_loss < first_loss * 0.3f && acc_after >= 0.8f;
  std::printf("test 2 (trains and improves in-batch retrieval accuracy): %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = test_gradient_check() && ok;
  ok = test_trains_and_improves_retrieval() && ok;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
