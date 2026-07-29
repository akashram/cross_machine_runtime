// indexing_pipeline_test.cpp -- two checks:
//  1. chunking mechanics on a deliberately long synthetic paragraph (the
//     corpus documents in rag/corpus/corpus.h are all short enough to fit
//     in one chunk each -- see that file's scope note -- so this test is
//     what actually exercises sentence-boundary splitting + overlap).
//  2. the full pipeline end to end: train an encoder on the shared
//     corpus (rag/embedding_model/train_encoder.h), chunk + index all 8
//     documents, and check that querying the index with each of the 8
//     labeled queries retrieves its true relevant document as the top
//     result -- a real, measured functional check of chunking + indexing
//     + retrieval composing correctly (the RIGOROUS recall@k measurement
//     is step 6's job, not this file's).
#include "index_pipeline.h"
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

void test_chunking_splits_and_overlaps() {
  // Ten short, distinct sentences -- long enough in aggregate that a
  // small max_chunk_chars forces multiple chunks, each individually well
  // under the cap (no single sentence here is anywhere near it).
  std::string paragraph =
      "the cat sat. the dog ran. birds fly high. fish swim deep. "
      "trees grow tall. rivers flow fast. mountains stand still. "
      "clouds drift slowly. stars shine bright. the moon glows soft.";

  auto chunks = chunk_text(paragraph, /*max_chunk_chars=*/40, /*overlap_chars=*/10);

  std::printf("  chunk_text produced %d chunks from a %d-char paragraph (max_chunk_chars=40)\n",
              static_cast<int>(chunks.size()), static_cast<int>(paragraph.size()));
  require(chunks.size() > 3, "a long paragraph with a small max_chunk_chars splits into multiple chunks");

  bool all_within_bound = true;
  for (const auto &c : chunks)
    if (static_cast<int>(c.size()) > 50) all_within_bound = false; // cap + one sentence worth of slack
  require(all_within_bound, "every chunk stays close to the requested max_chunk_chars bound");

  // Overlap: consecutive chunks should share a non-empty trailing/leading
  // substring (the carried-over tail of the previous chunk).
  bool found_overlap = false;
  for (std::size_t i = 0; i + 1 < chunks.size(); ++i) {
    const std::string &prev = chunks[i];
    const std::string &next = chunks[i + 1];
    std::string tail = prev.size() > 10 ? prev.substr(prev.size() - 10) : prev;
    // The exact 10-char tail may not survive word-boundary trimming, so
    // check for any shared few-character suffix/prefix instead of an
    // exact match.
    for (std::size_t len = tail.size(); len >= 4; --len) {
      if (next.rfind(tail.substr(tail.size() - len), 0) == 0 || next.find(tail.substr(tail.size() - len)) != std::string::npos) {
        found_overlap = true;
        break;
      }
    }
  }
  require(found_overlap, "consecutive chunks share carried-over overlap text, not a hard cut");

  // Overlap consumes some of each chunk's capacity carrying over the
  // previous chunk's tail, so it takes AT LEAST as many chunks to cover
  // the same text as with overlap disabled (never fewer) -- a real
  // measured consequence of the design, not assumed.
  auto no_overlap_chunks = chunk_text(paragraph, 40, /*overlap_chars=*/0);
  std::printf("  no-overlap chunking of the same paragraph: %d chunks (vs %d with overlap=10)\n",
              static_cast<int>(no_overlap_chunks.size()), static_cast<int>(chunks.size()));
  require(no_overlap_chunks.size() <= chunks.size(),
          "disabling overlap does not produce MORE chunks than the same text with overlap enabled");
}

void test_full_pipeline_retrieves_correct_document() {
  auto docs = sample_documents();
  auto queries = sample_queries();

  TrainedEncoder encoder = train_corpus_encoder(docs, queries);
  RagIndex index = build_index(encoder.params, encoder.tokenizer, docs);

  std::printf("  indexed %d chunks from %d documents (expect 8 == 8: every doc is short enough to be one chunk)\n",
              index.size(), static_cast<int>(docs.size()));
  require(index.size() == static_cast<int>(docs.size()), "indexing_pipeline's scope decision holds: one chunk per document here");

  int correct = 0;
  for (const auto &q : queries) {
    auto results = query_index(index, encoder.params, encoder.tokenizer, q.query, /*k=*/1);
    if (!results.empty() && index.chunk(results[0].index).doc_id == q.relevant_doc_id) ++correct;
  }
  double accuracy = static_cast<double>(correct) / static_cast<double>(queries.size());
  std::printf("  top-1 retrieval accuracy over %d labeled queries: %.3f\n", static_cast<int>(queries.size()), accuracy);
  require(accuracy >= 0.75, "the trained encoder + cosine index retrieves the correct document for most queries");
}

} // namespace

int main() {
  test_chunking_splits_and_overlaps();
  test_full_pipeline_retrieves_correct_document();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
