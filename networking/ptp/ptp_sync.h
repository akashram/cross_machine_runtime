#pragma once
#include <cstdint>
// TODO: implement on Linux with linuxptp

struct ClockSyncStats {
    double offset_mean_ns;
    double offset_stddev_ns;
    double max_error_ns;
};

// Query current offset from master clock via PTP
int64_t measure_offset_ns();

// Measure synchronization accuracy over N samples
ClockSyncStats measure_accuracy(int n_samples);
