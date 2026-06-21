#pragma once
// Memory coalescing validator — host-side only, no CUDA headers required.
//
// Runs ncu as a subprocess, parses its CSV output, and computes per-kernel
// sectors-per-request ratios for global loads and stores.
//
// Metric interpretation
// ---------------------
// l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  — 32-byte sectors fetched for loads
// l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum — warp-level load requests
//
// For a fully coalesced warp (32 threads × 4B = 128B = 4 sectors): ratio = 4.
// For fully uncoalesced stride-32 access (each thread in its own sector): ratio ≤ 32.
// NOTE: Calibrate threshold on your GPU after first run — card geometry
//       (sector size, L1 line size) affects the baseline ratio.

#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct CoalescingReport {
    std::string kernel_name;
    double load_sectors_per_request;   // lower is better; 4.0 = perfect for fp32
    double store_sectors_per_request;  // 0.0 when no stores issued
    bool pass;                         // true if both ≤ max_ratio (= 1/threshold × ideal)
};

namespace detail {

inline void rtrim(char* s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\r' || s[n-1] == '\n')) s[--n] = '\0';
}

// Parse one CSV field (quoted or bare); advances p past the trailing comma.
inline std::string csv_field(const char*& p) {
    std::string out;
    if (*p == '"') {
        ++p;
        while (*p) {
            if (*p == '"' && *(p+1) == '"') { out += '"'; p += 2; }  // escaped quote
            else if (*p == '"') { ++p; break; }
            else out += *p++;
        }
    } else {
        while (*p && *p != ',' && *p != '\r' && *p != '\n') out += *p++;
    }
    if (*p == ',') ++p;
    return out;
}

inline std::vector<std::string> csv_split(const char* line) {
    std::vector<std::string> fields;
    const char* p = line;
    while (*p && *p != '\r' && *p != '\n')
        fields.push_back(csv_field(p));
    return fields;
}

struct KernelAccum {
    double ld_sectors  = 0, ld_requests  = 0;
    double st_sectors  = 0, st_requests  = 0;
};

// Run ncu and collect per-kernel coalescing metric totals.
// Pass empty kernel_filter to capture all kernels.
inline std::map<std::string, KernelAccum>
run_ncu_coalescing(const std::string& binary, const std::string& kernel_filter) {
    static constexpr const char* METRICS =
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,"
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,"
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,"
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum";

    std::string cmd = "ncu --metrics " + std::string(METRICS);
    if (!kernel_filter.empty())
        cmd += " --kernel-name " + kernel_filter;
    // --csv writes structured CSV to stdout; ncu progress goes to stderr
    cmd += " --csv " + binary + " 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed: " + cmd);

    std::map<std::string, KernelAccum> accum;
    char line[8192];
    int col_kernel = -1, col_metric = -1, col_value = -1;
    bool header_done = false;

    while (fgets(line, sizeof(line), pipe)) {
        rtrim(line);
        if (line[0] == '\0') continue;

        auto fields = csv_split(line);
        if (fields.empty()) continue;

        if (!header_done) {
            for (int i = 0; i < (int)fields.size(); ++i) {
                if (fields[i] == "Kernel Name")  col_kernel = i;
                if (fields[i] == "Metric Name")  col_metric = i;
                if (fields[i] == "Metric Value") col_value  = i;
            }
            // Only mark header done once we find at least one expected column
            if (col_kernel >= 0 && col_metric >= 0 && col_value >= 0)
                header_done = true;
            continue;
        }

        int maxcol = std::max({col_kernel, col_metric, col_value});
        if ((int)fields.size() <= maxcol) continue;

        const std::string& kname  = fields[col_kernel];
        const std::string& mname  = fields[col_metric];
        const std::string& mval   = fields[col_value];
        if (kname.empty() || mval.empty()) continue;

        double val = 0;
        try { val = std::stod(mval); } catch (...) { continue; }

        auto& k = accum[kname];
        if      (mname == "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")  k.ld_sectors  += val;
        else if (mname == "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum") k.ld_requests += val;
        else if (mname == "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")  k.st_sectors  += val;
        else if (mname == "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum") k.st_requests += val;
    }
    pclose(pipe);
    return accum;
}

// threshold here is the fraction of ideal: 0.9 means sectors/request must be
// ≤ (ideal_ratio / threshold). Calibrate ideal_ratio from a known-good kernel.
inline CoalescingReport make_report(const std::string& name,
                                     const KernelAccum& k,
                                     double ideal_ratio,
                                     float threshold) {
    CoalescingReport r;
    r.kernel_name = name;
    r.load_sectors_per_request  = k.ld_requests > 0 ? k.ld_sectors  / k.ld_requests  : 0.0;
    r.store_sectors_per_request = k.st_requests > 0 ? k.st_sectors  / k.st_requests  : 0.0;
    double max_ratio = ideal_ratio / threshold;
    bool ld_ok = (k.ld_requests == 0) || (r.load_sectors_per_request  <= max_ratio);
    bool st_ok = (k.st_requests == 0) || (r.store_sectors_per_request <= max_ratio);
    r.pass = ld_ok && st_ok;
    return r;
}

} // namespace detail

// ideal_ratio: sectors/request for a perfectly coalesced kernel on the target GPU.
// For fp32 (4B) on Ampere/Turing/Volta with 32-byte sectors: typically 4.0.
// Calibrate with the coalesced_kernel in coalescing_test first, then set this.
static constexpr double kDefaultIdealRatio = 4.0;

inline CoalescingReport validate_kernel(const std::string& binary,
                                         const std::string& kernel_name,
                                         float threshold = 0.9f,
                                         double ideal_ratio = kDefaultIdealRatio) {
    auto accum = detail::run_ncu_coalescing(binary, kernel_name);
    auto it = accum.find(kernel_name);
    if (it == accum.end())
        throw std::runtime_error("kernel '" + kernel_name + "' not found in ncu output");
    return detail::make_report(kernel_name, it->second, ideal_ratio, threshold);
}

inline bool validate_all_kernels(const std::string& binary,
                                  float threshold,
                                  std::vector<CoalescingReport>& reports,
                                  double ideal_ratio = kDefaultIdealRatio) {
    auto accum = detail::run_ncu_coalescing(binary, "");
    bool all_pass = true;
    for (auto& [name, k] : accum) {
        auto r = detail::make_report(name, k, ideal_ratio, threshold);
        if (!r.pass) all_pass = false;
        reports.push_back(r);
    }
    return all_pass;
}
