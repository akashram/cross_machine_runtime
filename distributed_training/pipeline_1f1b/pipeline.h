#pragma once
// TODO: implement on multi-GPU

// 1F1B interleaved pipeline schedule.
// Reduces bubble fraction to 1/(m*p) vs GPipe's 1/p (m = microbatches, p = stages).
class PipelineSchedule {
public:
    PipelineSchedule(int num_stages, int num_microbatches, int stage_rank);

    // Run one iteration: interleave forward/backward across microbatches.
    void run_1f1b(std::function<void(int microbatch_id)> forward,
                  std::function<void(int microbatch_id)> backward);

    double bubble_fraction() const;
};
