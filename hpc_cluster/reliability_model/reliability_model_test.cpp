// reliability_model_test.cpp -- PLAN.md Phase 22 step 6 self-test.
//
// Every check is against a hand-computable closed-form value, same
// discipline as hyperband/ASHA's "re-verified against a hand-computable
// scenario with a known answer" (ml/hyperband/README.md) and pbt's
// non-decreasing-best invariant check.
#include "reliability_model.h"

#include <cmath>
#include <cstdio>

using namespace hpc_cluster;

namespace {
int g_failures = 0;
void check(bool cond, const char *what) {
  std::printf("%s  %s\n", cond ? "PASS" : "FAIL", what);
  if (!cond) ++g_failures;
}
bool close(double a, double b, double tol = 1e-9) { return std::fabs(a - b) < tol; }
} // namespace

int main() {
  // node_availability: MTBF=1000h, MTTR=10h -> 1000/1010 = 0.990099...
  {
    double a = node_availability(1000.0, 10.0);
    check(close(a, 1000.0 / 1010.0), "node_availability(1000,10) = 1000/1010");
  }

  // node_failure_rate_per_hour: MTBF=1000h -> 0.001/h
  {
    double r = node_failure_rate_per_hour(1000.0);
    check(close(r, 0.001), "node_failure_rate_per_hour(1000) = 0.001/h");
  }

  // cluster_first_failure_mtbf_hours: rates add for independent nodes --
  // 100 nodes at MTBF=100000h each -> cluster first-failure MTBF = 1000h.
  {
    double c = cluster_first_failure_mtbf_hours(100000.0, 100);
    check(close(c, 1000.0),
          "cluster_first_failure_mtbf_hours(100000, 100) = 1000h (rates add)");
  }

  // expected_failures_per_day: 100 nodes, MTBF=100000h ->
  // 100*24/100000 = 0.024 failures/day (~1 every ~41.7 days).
  {
    double f = expected_failures_per_day(100000.0, 100);
    check(close(f, 0.024), "expected_failures_per_day(100000, 100) = 0.024/day");
  }

  // cluster_availability_no_redundancy: p=0.99, n=3 -> 0.99^3 = 0.970299
  {
    double a = cluster_availability_no_redundancy(0.99, 3);
    check(close(a, 0.970299), "cluster_availability_no_redundancy(0.99,3) = 0.970299");
  }

  // n_choose_k sanity: C(5,2)=10, C(3,0)=1, C(3,3)=1
  {
    check(close(n_choose_k(5, 2), 10.0), "n_choose_k(5,2) = 10");
    check(close(n_choose_k(3, 0), 1.0), "n_choose_k(3,0) = 1");
    check(close(n_choose_k(3, 3), 1.0), "n_choose_k(3,3) = 1");
  }

  // k_of_n_availability, hand-computed: p=0.9, n=3, k=2 -- at least 2 of 3 up.
  // A = C(3,2)*0.9^2*0.1 + C(3,3)*0.9^3 = 3*0.81*0.1 + 0.729 = 0.243+0.729 = 0.972
  {
    double a = k_of_n_availability(0.9, 3, 2);
    check(close(a, 0.972, 1e-9), "k_of_n_availability(p=0.9,n=3,k=2) = 0.972 (hand-computed)");
  }

  // k_of_n_availability(p,n,n) must equal cluster_availability_no_redundancy
  // (all n of n up is the same condition, computed two structurally
  // different ways).
  {
    double a1 = k_of_n_availability(0.95, 5, 5);
    double a2 = cluster_availability_no_redundancy(0.95, 5);
    check(close(a1, a2), "k_of_n_availability(p,n,n) == cluster_availability_no_redundancy(p,n)");
  }

  // k_of_n_availability(p,n,1) is "at least one of n up" -- must be
  // 1 - (1-p)^n (complement of "all n down"), a second structurally
  // different closed form for the same k=1 case.
  {
    double p = 0.8;
    int n = 4;
    double a_binomial = k_of_n_availability(p, n, 1);
    double a_complement = 1.0 - std::pow(1.0 - p, n);
    check(close(a_binomial, a_complement, 1e-9),
          "k_of_n_availability(p,n,1) == 1-(1-p)^n (complement identity)");
  }

  // Redundancy tradeoff curve is monotonically non-decreasing in
  // availability as k decreases (more redundancy tolerance -> higher
  // availability), for a fixed n.
  {
    double p = 0.9;
    int n = 5;
    double prev = -1.0;
    bool monotonic = true;
    for (int k = n; k >= 1; --k) {
      double a = k_of_n_availability(p, n, k);
      if (a < prev - 1e-12) monotonic = false;
      prev = a;
    }
    check(monotonic,
          "k_of_n_availability is non-decreasing as k decreases (n=5, p=0.9) "
          "-- the redundancy/replication tradeoff curve");
  }

  std::printf(g_failures == 0 ? "\nALL PASS\n" : "\n%d FAILURES\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
