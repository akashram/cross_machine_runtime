#pragma once

// PLAN.md Phase 13 step 4 (indexing half): chunk every document (see
// chunking.h), embed each chunk with the step-1 encoder, and build a
// CosineBallTree (step 2) over the resulting vectors. Composition of
// already-built pieces, not new retrieval logic of its own.

#include "chunking.h"
#include "../../ml/knn/cosine_ann.h"
#include "../../transformer/char_tokenizer.h"
#include "../corpus/corpus.h"
#include "../embedding_model/embedding_model.h"

#include <string>
#include <vector>

namespace rag {

struct IndexedChunk {
  int doc_id;
  int chunk_id;
  std::string text;
};

inline std::vector<float> matrix_row_to_vector(const Matrix &m) {
  std::vector<float> v(static_cast<std::size_t>(m.cols()));
  for (int j = 0; j < m.cols(); ++j) v[static_cast<std::size_t>(j)] = m(0, j);
  return v;
}

// Move-only (CosineBallTree wraps BallTree, which owns a unique_ptr tree
// -- no copy semantics to preserve, same as BallTree/HNSW themselves).
class RagIndex {
public:
  RagIndex(std::vector<IndexedChunk> chunks, const Features &embeddings)
      : chunks_(std::move(chunks)), tree_(embeddings) {}

  std::vector<CosineNeighborResult> query(const std::vector<float> &query_embedding, int k,
                                           bool approximate = false) const {
    return tree_.query_knn(query_embedding, k, approximate);
  }

  const IndexedChunk &chunk(int i) const { return chunks_[static_cast<std::size_t>(i)]; }
  int size() const { return static_cast<int>(chunks_.size()); }

private:
  std::vector<IndexedChunk> chunks_;
  CosineBallTree tree_;
};

inline RagIndex build_index(const EncoderParams &encoder, const transformer::CharTokenizer &tok,
                             const std::vector<Document> &docs, int max_chunk_chars = 300, int overlap_chars = 30) {
  std::vector<IndexedChunk> chunks;
  Features embeddings;
  int next_chunk_id = 0;
  for (const auto &doc : docs) {
    for (const auto &piece : chunk_text(doc.text, max_chunk_chars, overlap_chars)) {
      chunks.push_back(IndexedChunk{doc.id, next_chunk_id++, piece});
      EncoderCache cache;
      Matrix emb = encode_forward(encoder, tok.encode(piece), cache);
      embeddings.push_back(matrix_row_to_vector(emb));
    }
  }
  return RagIndex(std::move(chunks), embeddings);
}

inline std::vector<CosineNeighborResult> query_index(const RagIndex &index, const EncoderParams &encoder,
                                                       const transformer::CharTokenizer &tok,
                                                       const std::string &query_text, int k,
                                                       bool approximate = false) {
  EncoderCache cache;
  Matrix emb = encode_forward(encoder, tok.encode(query_text), cache);
  return index.query(matrix_row_to_vector(emb), k, approximate);
}

} // namespace rag
