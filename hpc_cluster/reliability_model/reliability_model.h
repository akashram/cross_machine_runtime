//===- reliability_model.h - cluster MTBF/MTTR/availability model -------===//
//
// PLAN.md Phase 22 step 6: expected node failure rate, cluster-level
// availability given N nodes and per-node MTBF, and the
// redundancy/replication tradeoff curve.
//
// Standard reliability-engineering math (Trivedi, *Probability and
// Statistics with Reliability, Queuing, and Computer Science
// Applications*), not repo-specific invention -- the value this step adds
// is applying it directly to THIS repo's own cluster-design questions
// (how many nodes before "some node has failed" becomes the common case;
// what k-of-n redundancy buys), grounded partly in Schroeder & Gibson
// (2007)'s real large-scale HPC failure-rate study for a sense of scale
// (their data: annualized failure rates for real HPC/internet-service
// hardware commonly land in the 2-4%/year range per node, i.e. MTBF on
// the order of tens of thousands of hours -- used as the illustrative
// per-node MTBF constant in this step's own README, not a fabricated
// number).
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cmath>
#include <cstdint>

namespace hpc_cluster {

// Single-node availability: MTBF / (MTBF + MTTR) -- the fraction of time
// a node is expected to be up, given its mean time between failures and
// mean time to repair (Trivedi Ch. 8).
inline double node_availability(double mtbf_hours, double mttr_hours) {
  return mtbf_hours / (mtbf_hours + mttr_hours);
}

// Failure rate (failures/hour) for a single node: 1/MTBF. Standard
// exponential-lifetime reliability assumption (constant hazard rate) --
// the same assumption Trivedi's Ch. 8 MTBF-based availability formulas
// rest on.
inline double node_failure_rate_per_hour(double mtbf_hours) {
  return 1.0 / mtbf_hours;
}

// For N independent nodes each with the same per-node MTBF, failure
// RATES add (this is what "independent" means for exponential
// lifetimes): the cluster-level "time until the FIRST of N nodes fails"
// MTBF is node_mtbf / N. This is the number that matters for
// "how often do I need to touch this cluster at all," not the
// availability of any specific workload running on it.
inline double cluster_first_failure_mtbf_hours(double node_mtbf_hours, int n) {
  return node_mtbf_hours / static_cast<double>(n);
}

// Expected node failures per day across an N-node cluster.
inline double expected_failures_per_day(double node_mtbf_hours, int n) {
  return static_cast<double>(n) * 24.0 / node_mtbf_hours;
}

// Availability of an N-node cluster where ALL N nodes must be up
// simultaneously (no redundancy) -- e.g. a tightly-coupled MPI job that
// aborts if any single rank's node dies. Availability multiplies for
// independent components in series.
inline double cluster_availability_no_redundancy(double node_avail, int n) {
  return std::pow(node_avail, n);
}

inline double n_choose_k(int n, int k) {
  if (k < 0 || k > n) return 0.0;
  if (k == 0 || k == n) return 1.0;
  double result = 1.0;
  int kk = (k > n - k) ? n - k : k; // symmetry, fewer multiplications
  for (int i = 0; i < kk; ++i) {
    result *= static_cast<double>(n - i);
    result /= static_cast<double>(i + 1);
  }
  return result;
}

// Availability of a k-of-n system: at least k of n independent nodes
// (each with per-node availability p) must be up. Standard binomial
// reliability-block-diagram formula (Trivedi Ch. 8):
//   A(k,n) = sum_{i=k}^{n} C(n,i) * p^i * (1-p)^(n-i)
// k=n reduces to cluster_availability_no_redundancy (all must be up);
// k=1 is "at least one up" (N-way full replication) -- the two extremes
// of the redundancy/replication tradeoff curve this step is asked for.
inline double k_of_n_availability(double p, int n, int k) {
  double total = 0.0;
  for (int i = k; i <= n; ++i) {
    total += n_choose_k(n, i) * std::pow(p, i) * std::pow(1.0 - p, n - i);
  }
  return total;
}

} // namespace hpc_cluster
