#pragma once

// Character-level tokenizer: builds a vocabulary from the distinct bytes
// seen in a corpus, encode/decode between text and token ids. No BPE
// training pipeline — real, complete, and sufficient to make the
// transformer's cross-entropy loss and generation operate on real text
// rather than synthetic integers, without taking on the scope of a real
// subword tokenizer (a separate, much larger undertaking than what
// distributed_training/'s RLHF steps 22-25 need a model FOR).

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace transformer {

class CharTokenizer {
public:
  explicit CharTokenizer(const std::string &corpus) {
    for (char c : corpus) {
      auto byte = static_cast<uint8_t>(c);
      if (char_to_id_.find(byte) == char_to_id_.end()) {
        int id = static_cast<int>(id_to_char_.size());
        char_to_id_[byte] = id;
        id_to_char_.push_back(byte);
      }
    }
  }

  int vocab_size() const { return static_cast<int>(id_to_char_.size()); }

  std::vector<int> encode(const std::string &text) const {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (char c : text) {
      auto it = char_to_id_.find(static_cast<uint8_t>(c));
      if (it == char_to_id_.end()) throw std::runtime_error("CharTokenizer::encode: character not in vocabulary");
      ids.push_back(it->second);
    }
    return ids;
  }

  std::string decode(const std::vector<int> &ids) const {
    std::string text;
    text.reserve(ids.size());
    for (int id : ids) {
      if (id < 0 || id >= vocab_size()) throw std::runtime_error("CharTokenizer::decode: id out of range");
      text.push_back(static_cast<char>(id_to_char_[static_cast<size_t>(id)]));
    }
    return text;
  }

private:
  std::unordered_map<uint8_t, int> char_to_id_;
  std::vector<uint8_t> id_to_char_;
};

} // namespace transformer
