// rag_generation_test.cpp -- two checks:
//  1. construct_prompt() mechanics: retrieved chunk text and the query
//     both land in the constructed prompt, in the expected structure, and
//     the no-retrieval case degrades to a query-only prompt (the shape
//     step 7's with-vs-without-retrieval ablation needs).
//  2. rag_generate() wiring: retrieval -> prompt -> causal generation
//     actually composes -- exactly max_new_tokens new tokens come back,
//     they decode to a valid string, and retrieval measurably changes
//     what gets fed to the model (the prompt differs with vs. without
//     it). Deliberately uses UNTRAINED encoder/causal-model weights: this
//     step verifies the plumbing, not generation quality (step 7's job,
//     with a causal model actually trained for the answer-from-context
//     skill) or retrieval quality (steps 4/6's job).
#include "rag_generate.h"
#include "../corpus/corpus.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>

using namespace rag;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_construct_prompt_structure() {
  std::vector<std::string> chunks{"paris is the capital of france.", "it has the eiffel tower."};
  std::string prompt = construct_prompt(chunks, "what is the capital of france");

  bool has_chunk0 = prompt.find(chunks[0]) != std::string::npos;
  bool has_chunk1 = prompt.find(chunks[1]) != std::string::npos;
  bool has_query = prompt.find("what is the capital of france") != std::string::npos;
  bool query_after_context = prompt.find(chunks[0]) < prompt.find("what is the capital of france");

  require(has_chunk0 && has_chunk1, "constructed prompt contains every retrieved chunk's text");
  require(has_query, "constructed prompt contains the query text");
  require(query_after_context, "retrieved context precedes the question in the constructed prompt");

  std::string no_context_prompt = construct_prompt({}, "what is the capital of france");
  bool no_context_has_query = no_context_prompt.find("what is the capital of france") != std::string::npos;
  bool no_context_has_chunk = no_context_prompt.find(chunks[0]) != std::string::npos;
  require(no_context_has_query && !no_context_has_chunk,
          "with no retrieved chunks, the prompt degrades to query-only (no leftover context text)");
}

void test_rag_generate_wiring() {
  auto docs = sample_documents();
  auto queries = sample_queries();

  // Untrained encoder: this test checks WIRING, not retrieval quality
  // (already measured in indexing_pipeline/recall_eval).
  std::string joint;
  for (const auto &d : docs) joint += d.text + " ";
  for (const auto &q : queries) joint += q.query + " ";
  transformer::CharTokenizer encoder_tok(joint);
  EncoderConfig enc_cfg{encoder_tok.vocab_size(), /*d_model=*/8, /*num_heads=*/2, /*num_layers=*/1, /*d_ff=*/16,
                        /*max_seq_len=*/128, /*embed_dim=*/6};
  std::mt19937 enc_rng(3);
  EncoderParams encoder = init_encoder(enc_cfg, enc_rng);
  RagIndex index = build_index(encoder, encoder_tok, docs);

  // Untrained causal model: same reasoning. Its tokenizer's vocab must
  // cover every character that can appear in a constructed prompt,
  // including the "Context:"/"Question:"/"Answer:" template literals.
  std::string causal_corpus = joint + "Context: Question: Answer: ";
  transformer::CharTokenizer causal_tok(causal_corpus);
  transformer::TransformerConfig causal_cfg{causal_tok.vocab_size(), /*d_model=*/8, /*num_heads=*/2,
                                             /*num_layers=*/1, /*d_ff=*/16, /*max_seq_len=*/256};
  std::mt19937 causal_rng(4);
  transformer::ModelParams causal_model = transformer::init_model(causal_cfg, causal_rng);

  const std::string &query = queries[0].query;
  constexpr int kMaxNewTokens = 10, kK = 2;

  auto with_retrieval = rag_generate(index, encoder, encoder_tok, causal_model, causal_tok, query, kK, kMaxNewTokens,
                                      /*use_retrieval=*/true);
  auto without_retrieval = rag_generate(index, encoder, encoder_tok, causal_model, causal_tok, query, kK,
                                         kMaxNewTokens, /*use_retrieval=*/false);

  std::printf("  with retrieval:    %d chunk(s), prompt length=%d, continuation length=%d\n",
              static_cast<int>(with_retrieval.retrieved_chunk_texts.size()), static_cast<int>(with_retrieval.prompt.size()),
              static_cast<int>(with_retrieval.continuation_text.size()));
  std::printf("  without retrieval: %d chunk(s), prompt length=%d, continuation length=%d\n",
              static_cast<int>(without_retrieval.retrieved_chunk_texts.size()),
              static_cast<int>(without_retrieval.prompt.size()), static_cast<int>(without_retrieval.continuation_text.size()));

  require(static_cast<int>(with_retrieval.continuation_text.size()) == kMaxNewTokens,
          "rag_generate produces exactly max_new_tokens new characters (char-level, 1 token = 1 char)");
  require(with_retrieval.retrieved_chunk_texts.size() == static_cast<std::size_t>(kK),
          "rag_generate retrieves k chunks when use_retrieval=true");
  require(without_retrieval.retrieved_chunk_texts.empty(), "rag_generate retrieves nothing when use_retrieval=false");
  require(with_retrieval.prompt.size() > without_retrieval.prompt.size(),
          "retrieval measurably changes what gets fed to the causal model (longer, context-bearing prompt)");
  require(with_retrieval.prompt != without_retrieval.prompt,
          "the with-retrieval and without-retrieval prompts are not identical");
}

} // namespace

int main() {
  test_construct_prompt_structure();
  test_rag_generate_wiring();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
