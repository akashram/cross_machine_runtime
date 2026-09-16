//===- storage_comparison.h - VAST vs. WekaFS vs. Lustre/GPFS vs. Ceph --===//
//
// PLAN.md Phase 21 step 3: a parallel/distributed storage system
// comparison for AI training workloads. No rentable access to any of
// these four systems exists (same disclosed limitation as
// analog_engine/nvm_comparison's device comparison), so every field below
// is a literature/vendor-documentation-grounded REPRESENTATIVE
// characterization, not a measurement -- honestly labeled, same
// convention as PLAN.md explicitly asks for ("same honest-labeling
// convention as Phase 17's NVM comparison").
//
// Sources (see hpc_storage/DESIGN.md and READING_LIST.md's Phase 21
// section for full citations): VAST Data's published DASE architecture
// whitepapers, WekaFS's published architecture documentation, Lustre's
// and GPFS's well-documented OSS/OST and NSD architectures (both have
// 20+ years of published HPC-center deployment literature), Ceph's RADOS
// architecture papers (Weil et al.) and its well-known AI-workload
// metadata-latency characteristics vs. purpose-built parallel filesystems.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace hpc_storage {

struct ParallelFsClass {
  std::string name;
  std::string architecture;         // one-line architecture summary
  // 1 (worst) .. 5 (best), a representative qualitative rating for
  // AI-training-relevant small-file/metadata throughput -- the axis
  // step 2's local-filesystem study demonstrates the mechanism for.
  int small_file_metadata_rating;
  bool gds_certified;               // NVIDIA GPUDirect Storage certification, publicly listed
  std::string data_reduction_technique;
  std::string typical_deployment;
  std::string note;
};

inline const std::array<ParallelFsClass, 4> &parallel_fs_classes() {
  static const std::array<ParallelFsClass, 4> table = {{
      // VAST Data: Disaggregated Shared Everything (DASE). CNodes
      // (stateless protocol/compute) fully decoupled from DNodes (QLC
      // flash + NVRAM write buffering), global namespace, similarity-based
      // ("VAST Global Reduce") data reduction across the entire cluster
      // rather than per-node. GPUDirect Storage certified. Purpose-built
      // for AI/ML training at hyperscale, commonly cited in MLPerf Storage
      // submissions.
      {"VAST Data", "Disaggregated Shared Everything (DASE): stateless CNodes + QLC-flash/NVRAM DNodes, global namespace",
       5, true, "Global similarity-based reduction (cluster-wide, not per-node)",
       "Hyperscale AI training clusters; vendor-managed all-flash appliance",
       "No hard disqualifier on any AI-workload axis; the disaggregation is specifically what lets CNode "
       "(protocol) and DNode (media) capacity scale independently, unlike traditional OSS/OST coupling."},
      // WekaFS: POSIX-compliant distributed filesystem, NVMe-oF-based,
      // also GDS-certified, strong small-file/metadata performance (a
      // frequently cited differentiator vs. Lustre/GPFS in published
      // MLPerf Storage results), software-defined (runs on commodity
      // NVMe rather than a fixed appliance).
      {"WekaFS", "Distributed POSIX filesystem over NVMe-oF, software-defined (commodity or appliance)", 5, true,
       "Per-cluster compression + data reduction",
       "AI/ML and HPC clusters wanting software-defined flexibility over VAST's fixed appliance model",
       "Matches VAST on small-file/GDS-certification axes; the real differentiator vs. VAST is deployment "
       "model (software-defined on your own NVMe vs. a fixed appliance), not raw performance."},
      // Lustre / GPFS (grouped: both are the classic HPC parallel
      // filesystem architecture -- OSS/OST for Lustre, NSD for GPFS/IBM
      // Storage Scale). Centralized metadata server (Lustre's MDS; GPFS
      // distributes metadata more than Lustre but both are architecturally
      // older than VAST/WekaFS's fully disaggregated design). Extremely
      // well proven for large-sequential HPC scientific I/O; the MDS
      // single point of contention is a well-documented real bottleneck
      // for the small-file-heavy, metadata-op-heavy pattern AI training's
      // shuffled dataset loading produces (exactly the mechanism step 2
      // measures locally).
      {"Lustre/GPFS", "Classic parallel FS: OSS/OST (Lustre) or NSD (GPFS) object/block servers + a metadata "
                       "server tier",
       2, false, "Filesystem/hardware-level compression (not workload-aware global dedup)",
       "Traditional HPC centers; decades of large-sequential scientific-simulation I/O tuning",
       "The most mature and widely deployed of the four for classic HPC I/O, but its metadata-server tier "
       "(Lustre's MDS especially) is a well-documented bottleneck for AI training's small-file/shuffled-read "
       "pattern -- the opposite I/O shape from the large-sequential scientific writes it was designed for."},
      // Ceph: software-defined, CRUSH-algorithm placement, RADOS object
      // store underneath CephFS/RBD/RGW. Extremely flexible and
      // commodity-hardware-friendly, but its general-purpose object-store
      // design trades away the AI-training-specific metadata/small-file
      // optimizations VAST/WekaFS built for, and it lacks a mainstream
      // GDS certification. Included as the open-source/self-hosted
      // baseline -- and the system step 11's own hands-on tuning falls
      // back to only if MinIO isn't chosen.
      {"Ceph", "Software-defined RADOS object store (CRUSH placement) underneath CephFS/RBD/RGW", 3, false,
       "None built-in at the RADOS layer (relies on underlying block/OSD storage)",
       "Self-hosted, commodity hardware, general-purpose (block+object+file) rather than AI-training-specific",
       "The most flexible and cheapest of the four to self-host, but general-purpose by design -- not "
       "purpose-optimized for AI training's specific I/O shape the way VAST/WekaFS are, and no mainstream "
       "GDS certification as of this writing."},
  }};
  return table;
}

struct AiWorkloadScore {
  std::string name;
  double score; // 0..1, illustrative composite -- see compute_ai_workload_score()'s doc comment
};

// An explicitly-illustrative composite "AI training workload fit" score:
// mean of (small_file_metadata_rating/5, gds_certified as 1.0/0.0), NOT a
// validated procurement decision model -- the same illustrative-scoring
// convention as analog_engine/nvm_comparison::compute_figure_of_merit(),
// stated for the same reason: reducing genuinely different axes to one
// ranking number is useful for a quick comparison, not a substitute for
// an actual bake-off on real workload data.
inline std::vector<AiWorkloadScore> compute_ai_workload_score() {
  const auto &table = parallel_fs_classes();
  std::vector<AiWorkloadScore> out;
  for (const auto &fs : table) {
    double metadata_norm = static_cast<double>(fs.small_file_metadata_rating) / 5.0;
    double gds_norm = fs.gds_certified ? 1.0 : 0.0;
    double score = (metadata_norm + gds_norm) / 2.0;
    out.push_back({fs.name, score});
  }
  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.score > b.score; });
  return out;
}

} // namespace hpc_storage
