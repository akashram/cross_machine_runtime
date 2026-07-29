#pragma once

// PLAN.md Phase 13 step 4 (chunking half): sentence-boundary-aware
// chunking with overlap -- split at '.', '!', '?' boundaries (never
// mid-sentence), then greedily pack sentences into chunks up to
// `max_chunk_chars`, carrying the previous chunk's trailing
// `overlap_chars` characters into the next chunk's start (so a fact
// straddling a chunk boundary is still fully present in at least one
// chunk).

#include <string>
#include <vector>

namespace rag {

inline std::vector<std::string> split_sentences(const std::string &text) {
  std::vector<std::string> sentences;
  std::string current;
  std::size_t i = 0;
  while (i < text.size()) {
    char c = text[i];
    current += c;
    bool is_boundary = (c == '.' || c == '!' || c == '?');
    bool at_end = (i + 1 == text.size());
    bool next_is_space = !at_end && text[i + 1] == ' ';
    if (is_boundary && (at_end || next_is_space)) {
      std::size_t start = current.find_first_not_of(' ');
      if (start != std::string::npos) sentences.push_back(current.substr(start));
      current.clear();
      i += next_is_space ? 2 : 1; // also skip the following space
      continue;
    }
    ++i;
  }
  if (!current.empty()) {
    std::size_t start = current.find_first_not_of(' ');
    if (start != std::string::npos) sentences.push_back(current.substr(start));
  }
  return sentences;
}

// Never splits a sentence in half -- a single sentence longer than
// max_chunk_chars becomes its own (oversized) chunk rather than being cut
// mid-word; not a concern for any document in rag/corpus/corpus.h, but a
// real, disclosed scope boundary rather than silently doing the wrong
// thing on hypothetical longer input.
inline std::vector<std::string> chunk_text(const std::string &text, int max_chunk_chars, int overlap_chars) {
  std::vector<std::string> sentences = split_sentences(text);
  std::vector<std::string> chunks;
  std::string current;
  for (const auto &s : sentences) {
    if (!current.empty() && static_cast<int>(current.size() + 1 + s.size()) > max_chunk_chars) {
      chunks.push_back(current);
      if (overlap_chars > 0 && static_cast<int>(current.size()) > overlap_chars)
        current = current.substr(current.size() - static_cast<std::size_t>(overlap_chars));
      else
        current.clear();
    }
    if (!current.empty()) current += " ";
    current += s;
  }
  if (!current.empty()) chunks.push_back(current);
  return chunks;
}

} // namespace rag
