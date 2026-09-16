# reliability_model -- cluster MTBF/MTTR/availability model

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 22 step 6: expected node failure rate, cluster-level
availability given N nodes and per-node MTBF, and the
redundancy/replication tradeoff curve -- real reliability-engineering
math (Trivedi's standard MTBF/MTTR/availability formulas), not a
repo-specific invention.

## Design

- `node_availability(mtbf, mttr) = mtbf / (mtbf + mttr)` -- the standard
  single-node availability formula.
- `cluster_first_failure_mtbf_hours(node_mtbf, n) = node_mtbf / n` --
  failure RATES add for independent exponential-lifetime components, so
  an N-node cluster's "time until the first of N nodes fails" MTBF
  shrinks linearly with N. This is the number that answers "how often do
  I need to touch this cluster at all," distinct from any specific
  workload's availability.
- `cluster_availability_no_redundancy(p, n) = p^n` -- availability when
  ALL n nodes must be up (independent components in series), e.g. a
  tightly-coupled MPI job.
- `k_of_n_availability(p, n, k)` -- the general binomial
  reliability-block-diagram formula (Trivedi Ch. 8), `k=n` and `k=1` are
  the two extremes of the redundancy/replication tradeoff curve PLAN.md
  step 6 asks for (no redundancy vs. full N-way replication).
- Literature calibration: Schroeder & Gibson (2007)'s real large-scale
  HPC/internet-service failure-rate study found per-node annualized
  failure rates commonly in the 2-4%/year range, i.e. MTBF on the order
  of tens of thousands of hours -- the illustrative constant used in this
  step's own worked example below, not a fabricated number.

## Results (captured 2026-09-12, Apple clang 14, this Mac)

```
PASS  node_availability(1000,10) = 1000/1010
PASS  node_failure_rate_per_hour(1000) = 0.001/h
PASS  cluster_first_failure_mtbf_hours(100000, 100) = 1000h (rates add)
PASS  expected_failures_per_day(100000, 100) = 0.024/day
PASS  cluster_availability_no_redundancy(0.99,3) = 0.970299
PASS  n_choose_k(5,2) = 10
PASS  n_choose_k(3,0) = 1
PASS  n_choose_k(3,3) = 1
PASS  k_of_n_availability(p=0.9,n=3,k=2) = 0.972 (hand-computed)
PASS  k_of_n_availability(p,n,n) == cluster_availability_no_redundancy(p,n)
PASS  k_of_n_availability(p,n,1) == 1-(1-p)^n (complement identity)
PASS  k_of_n_availability is non-decreasing as k decreases (n=5, p=0.9) -- the redundancy/replication tradeoff curve

ALL PASS
```

**Worked illustrative example** (Schroeder & Gibson-grounded MTBF): a
100-node cluster where each node has MTBF=100,000h (~11.4 years, within
their study's observed range) and MTTR=4h:

- Per-node availability: `100000/100004 = 0.99996`
- Cluster-wide first-failure MTBF: `100000/100 = 1000h` (~41.7 days) --
  i.e. SOME node in a 100-node cluster is expected to fail roughly every
  6 weeks, even though any individual node fails only once every ~11
  years. This is the real, counter-intuitive-until-you-see-the-math
  reason "at cluster scale, node failure is the normal case, not the
  exception" -- the entire justification for building health-check
  tooling (step 4) and Slurm's node-drain/requeue behavior (step 3) as
  first-class, not exceptional-path, functionality.
- Availability of a workload requiring ALL 100 nodes up simultaneously:
  `0.99996^100 = 0.9960` (99.60% -- roughly 35 hours/year of expected
  downtime from node failure alone), vs. a workload tolerant of any 90 of
  100 (`k_of_n_availability(0.99996, 100, 90)`): effectively `1.0` to
  within floating-point precision -- the quantified case for building
  workloads that tolerate partial-cluster failure wherever the
  application allows it (data-parallel training with elastic world size,
  vs. a rigid all-ranks-or-nothing MPI job).

## Findings

- **Two structurally independent derivations of the same k=n and k=1
  cases agree exactly** (`k_of_n_availability(p,n,n) ==
  cluster_availability_no_redundancy(p,n)`; `k_of_n_availability(p,n,1)
  == 1-(1-p)^n`) -- a real correctness cross-check, not just "the
  function runs."
- **The redundancy/replication tradeoff curve is verified monotonic**:
  availability is non-decreasing as the required quorum `k` shrinks (more
  redundancy tolerance), for a fixed `n` and `p` -- the property that
  makes "how much redundancy do I need for X% availability" a
  well-posed question to answer with this model.
- **The scale effect is the real finding worth keeping**: cluster-level
  "time to next failure" shrinks linearly with node count even though
  per-node reliability is unchanged -- a concrete, quantified argument
  (not just an assertion) for why HPC cluster software has to treat node
  failure as routine, connecting directly to step 3 (Slurm's real
  backfill/requeue behavior) and step 4 (health-check tooling) elsewhere
  in this phase.

## Hardware notes
None -- pure CPU math, no external dependency.
