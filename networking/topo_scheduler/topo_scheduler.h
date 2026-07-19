//===- topo_scheduler.h - Topology-aware task placement -------------------===//
//
// Portable — no nvml/PCIe/sysfs dependency. Real topology *discovery*
// (PCIe tree via /sys, NVLink topology via nvml, EFA bandwidth map, FPGA
// attachment paths — PLAN.md's list for this step) is Linux/GPU-hardware
// specific and out of scope for this file; it would populate a
// TopologyGraph the same way this header's test data does by hand. The
// *placement algorithm* — given a topology and a workload's communication
// pattern, decide which task runs where — has nothing hardware-specific
// about it, so it's tested here without that discovery code existing yet.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace topo {

struct Edge {
  int to;
  double bandwidth_gbs;
};

// Adjacency-list graph: node i's neighbors are links[i]. Bandwidth-only
// cost model (no separate latency term) — good enough for placement
// decisions dominated by bulk collective traffic (this project's actual
// use case: placing communicating tensor-parallel ranks, MoE experts,
// pipeline stages).
struct TopologyGraph {
  std::vector<std::string> nodeNames;
  std::vector<std::vector<Edge>> links;

  int addNode(const std::string &name);
  void addLink(int a, int b, double bandwidth_gbs); // undirected
};

// Task i communicates `volume[i][j]` GB with task j over the course of
// the workload (symmetric; only the sum i<j is used). place() decides,
// for each task, which topology node it runs on.
struct Workload {
  int numTasks;
  std::vector<std::vector<double>> commVolumeGb; // numTasks x numTasks
};

// Greedy placement: process task pairs in descending communication-volume
// order; each task, when first encountered, is assigned to whichever free
// topology node minimizes total (volume / bandwidth) "transfer time" cost
// to its already-placed communication partners, using all-pairs
// bandwidth-weighted shortest paths precomputed once. Returns
// task index -> topology node index.
//
// Not globally optimal (this is graph partitioning, NP-hard in general) —
// same "greedy, documented, revisit once there's a real workload to
// benchmark the alternative against" stance as the placement pass in
// Phase 4 (compiler/placement, step 9). See networking/DESIGN.md.
std::vector<int> place(const TopologyGraph &topology, const Workload &workload);

} // namespace topo
