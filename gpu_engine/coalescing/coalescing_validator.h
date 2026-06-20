#pragma once
#include <string>
#include <vector>

// TODO: implement on GPU hardware using ncu subprocess + CSV parsing

struct CoalescingReport {
    std::string kernel_name;
    double load_sectors_per_request;   // ideal = 1.0
    double store_sectors_per_request;  // ideal = 1.0
    bool pass;                         // true if both ≤ 1/threshold
};

// Run ncu on binary, extract coalescing metrics for kernel_name.
// threshold: minimum coalescing ratio (0.0–1.0); default 0.9 → sectors/req ≤ 1.11
CoalescingReport validate_kernel(const std::string& binary,
                                  const std::string& kernel_name,
                                  float threshold = 0.9f);

// Validate all kernels in binary; returns false if any fail.
bool validate_all_kernels(const std::string& binary,
                           float threshold,
                           std::vector<CoalescingReport>& reports);
