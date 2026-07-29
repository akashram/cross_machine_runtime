// serving_integration_test.cpp -- PLAN.md Phase 13 step 9: does the RAG
// request path actually go through ServingRouter's real backend
// dispatch/fallback logic (Phase 9 step 8), not a hand-rolled generation
// call of its own? Same "verify the wiring, not quality" scope as step
// 5's rag_generation_test.cpp -- untrained encoder/causal-model weights,
// since generation quality is already covered by step 7 and retrieval
// quality by steps 4/6.
#include "rag_serving.h"
#include "../corpus/corpus.h"

#include <cstdio>
#include <random>
#include <string>

using namespace rag;
using inference_serving::Backend;
using inference_serving::ServingRouter;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

ServingRouter make_cpu_only_router() {
  ServingRouter router;
  router.register_backend(Backend::GPU, {false, "CUDA not found"});
  router.register_backend(Backend::FPGA, {false, "XILINX_VITIS not set"});
  router.register_backend(Backend::TPU, {false, "no TPU device"});
  router.register_backend(Backend::CPU, {true, ""}, inference_serving::make_cpu_backend());
  return router;
}

} // namespace

int main() {
  auto docs = sample_documents();
  auto queries = sample_queries();

  std::string joint;
  for (const auto &d : docs) joint += d.text + " ";
  for (const auto &q : queries) joint += q.query + " ";
  transformer::CharTokenizer encoder_tok(joint);
  EncoderConfig enc_cfg{encoder_tok.vocab_size(), /*d_model=*/8, /*num_heads=*/2, /*num_layers=*/1, /*d_ff=*/16,
                        /*max_seq_len=*/128, /*embed_dim=*/6};
  std::mt19937 enc_rng(6);
  EncoderParams encoder = init_encoder(enc_cfg, enc_rng);
  RagIndex index = build_index(encoder, encoder_tok, docs);

  std::string causal_corpus = joint + "Context: Question: Answer: ";
  transformer::CharTokenizer causal_tok(causal_corpus);
  transformer::TransformerConfig causal_cfg{causal_tok.vocab_size(), /*d_model=*/8, /*num_heads=*/2,
                                             /*num_layers=*/1, /*d_ff=*/16, /*max_seq_len=*/256};
  std::mt19937 causal_rng(7);
  transformer::ModelParams causal_model = transformer::init_model(causal_cfg, causal_rng);

  ServingRouter router = make_cpu_only_router();
  const std::string &query = queries[0].query;
  constexpr int kK = 2, kMaxNewTokens = 8;

  auto result = route_rag(router, Backend::GPU, index, encoder, encoder_tok, causal_model, causal_tok, query, kK,
                           kMaxNewTokens);

  std::printf("  routed via GPU (unavailable): backend_used=%s, fell_back=%s, retrieved %d chunk(s)\n",
              inference_serving::to_string(result.backend_used), result.fell_back ? "true" : "false",
              static_cast<int>(result.retrieved_chunk_texts.size()));

  require(result.backend_used == Backend::CPU, "requesting an unavailable backend (GPU) falls back to CPU through the SAME ServingRouter fallback logic Phase 9 already tests");
  require(result.fell_back, "route_rag reports that a fallback occurred");
  require(result.retrieved_chunk_texts.size() == static_cast<std::size_t>(kK), "route_rag retrieves k chunks before dispatching to the router");
  require(result.prompt.find(result.retrieved_chunk_texts[0]) != std::string::npos,
          "the retrieved chunk text is actually present in the prompt handed to the router");
  require(static_cast<int>(result.continuation_text.size()) == kMaxNewTokens,
          "the router-dispatched generation produces exactly max_new_tokens new characters");

  auto direct_result = route_rag(router, Backend::CPU, index, encoder, encoder_tok, causal_model, causal_tok, query,
                                  kK, kMaxNewTokens);
  require(!direct_result.fell_back, "requesting the available CPU backend directly needs no fallback");
  require(direct_result.full_tokens == result.full_tokens,
          "GPU-preferred-with-fallback and CPU-preferred-directly reach the identical CPU backend and produce identical output");

  auto no_retrieval_result = route_rag(router, Backend::CPU, index, encoder, encoder_tok, causal_model, causal_tok,
                                        query, kK, kMaxNewTokens, /*use_retrieval=*/false);
  require(no_retrieval_result.retrieved_chunk_texts.empty(), "use_retrieval=false skips retrieval entirely, same as rag_generate()");
  require(no_retrieval_result.prompt != result.prompt, "the no-retrieval request path produces a different (shorter, context-free) prompt");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
