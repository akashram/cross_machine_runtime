#pragma once
#include <vector>
#include <unordered_map>
// TODO: implement on GPU — PagedAttention (Kwon et al., vLLM)

struct KVTensor {
    void*  data;        // device pointer (K or V, one head)
    int    head_dim;
    int    seq_len;
};

struct BlockTable {
    std::vector<int> physical_blocks;  // maps logical block → physical block id
};

class PagedKVCache {
public:
    PagedKVCache(int num_layers, int num_heads, int head_dim,
                 int block_size = 16, int max_blocks = 2048);

    // Append one token's KV to a sequence (allocates block if needed)
    void append(int seq_id, int layer, const KVTensor& k, const KVTensor& v);

    // Get the block table for a sequence (used by attention kernel)
    const BlockTable& get_block_table(int seq_id) const;

    // Free all blocks for a completed sequence
    void free(int seq_id);

    int num_free_blocks() const;

private:
    // TODO: free block list, block allocator, sequence block tables
    std::unordered_map<int, BlockTable> seq_tables_;
};
