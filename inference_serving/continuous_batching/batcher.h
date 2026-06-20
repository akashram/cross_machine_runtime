#pragma once
#include <vector>
#include <functional>
// TODO: implement on GPU

struct Request {
    int         id;
    std::string prompt;
    int         max_new_tokens;
    double      arrival_time_s;
};

struct Batch {
    std::vector<int>    seq_ids;
    std::vector<int*>   input_ids;    // device pointers
    std::vector<int>    seq_lengths;
};

class ContinuousBatcher {
public:
    void add_request(Request r);

    // Form next batch from waiting requests and active sequences.
    // Returns empty batch if nothing to run.
    Batch next_batch(int max_batch_size = 32);

    // Mark sequence as finished (frees KV cache blocks).
    void finish_sequence(int seq_id);
};
