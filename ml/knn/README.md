# knn

**Status: code-complete AND locally run — pure CPU, no external
dependency. The `vs sklearn on OpenML CC-18` comparison stays TODO —
deferred pending the install decision covering that benchmark suite
(see project memory); real correctness/parameter-effect results below
don't depend on it.**

## What this measures

PLAN.md Phase 12a step 5: k-NN via a KD-tree (exact) and a ball tree
(approximate), SIMD distance computation.

This directory also holds PLAN.md **Phase 13 step 2** (`cosine_ann.h/.cpp`,
`cosine_ann_test.cpp`): cosine-similarity ANN retrieval for
`rag/`'s embedding vectors, built as a thin wrapper around this file's own
`BallTree` — see the dedicated "cosine_ann" section below.

## Design

- `squared_distance()`: a single linear pass over contiguous float
  arrays — auto-vectorizable at `-O3`, same convention as
  `decision_tree`/`svm`'s "no hand-written AVX intrinsics" honest caveat
  (see their READMEs' Design notes).
- **KDTree**: classic depth-round-robin-split (Bentley 1975), one point
  per node, branch-and-bound pruning — the far child is only visited if
  the query's distance to the splitting hyperplane could still beat the
  current k-th best. Always exact.
- **BallTree**: two-pivot construction (Omohundro 1989 / Uhlmann 1991) —
  pick the point farthest from the subtree centroid (`p1`), then the
  point farthest from `p1` (`p2`), partition by nearer pivot; degenerate
  partitions (e.g. every point on one side) fall back to a plain
  index-range median split so recursion always makes progress. Leaves
  hold up to `leaf_size` points (default 10). Queryable in two modes
  against the **same tree structure**, so they're directly comparable:
  - `approximate=false`: bound-and-backtrack using the triangle-inequality
    radius bound (`dist(query,center) - radius`) to prune whole subtrees.
    Same correctness guarantee as KDTree, verified against it directly
    (see Results).
  - `approximate=true`: **defeatist search** — descend straight to the
    nearer child at every internal node, never visit the sibling
    subtree, brute-force the leaf reached. `O(log n)` node visits, not
    guaranteed exact. This is PLAN.md's "ball tree (approximate)".
- `KNNClassifier` wraps either structure behind one interface
  (`KNNParams::structure`), majority-vote prediction with ties broken by
  the single nearest neighbor among tied classes.

## Results (captured 2026-07-28, Apple clang 14 / `-std=c++2b`, this Mac)

```
  KDTree: 0/20 queries mismatched brute force; avg nodes visited=118.3 / 300 points
PASS  KDTree's branch-and-bound pruning finds the same k nearest neighbors as brute force
PASS  KDTree visits fewer nodes than a full brute-force scan (pruning actually prunes)
  BallTree (exact): 0/20 queries mismatched brute force
PASS  BallTree's exact (bound-and-backtrack) mode finds the same k nearest neighbors as brute force
  BallTree defeatist: recall=0.340 vs exact, avg nodes visited exact=15.9 approx=6.2
PASS  defeatist search actually misses some true neighbors on this data (a genuine approximation, not accidentally exact)
PASS  defeatist search still finds a nontrivial fraction of true neighbors (not landing in an unrelated cluster)
PASS  defeatist search visits fewer nodes than exact backtracking search (the speed side of the tradeoff)
  held-out accuracy: k=1 -> 0.758, k=15 -> 1.000
PASS  a larger k generalizes at least as well as k=1 on noisy-label training data (majority vote out-votes individual noisy neighbors)
  well-separated blobs train accuracy: KD_TREE=1.000 BALL_TREE(exact)=1.000
PASS  KDTree and exact BallTree produce identical classification accuracy (same exact-neighbor semantics)
PASS  k-NN separates two well-separated blobs almost exactly
PASS
```

## Findings

- Both KDTree and BallTree's exact mode match a brute-force O(n) scan on
  every one of 40 test queries — the branch-and-bound / radius-bound
  pruning never changes the answer, only how many nodes get visited
  (118.3/300 for KDTree in 5D; 15.9/~300 for BallTree's tighter radius
  bound on 16D clustered data).
- Defeatist ball-tree search is a genuine approximation, not an
  accidentally-exact shortcut: on 5 well-separated 16D Gaussian clusters
  with in-distribution queries (a new sample near a cluster center, the
  realistic k-NN query pattern — an out-of-distribution query in empty
  space makes "nearest neighbor" close to meaningless and was tried
  first, producing near-zero recall for both structures' answers being
  arbitrary far-away points), defeatist search visits 6.2 nodes on
  average vs. exact's 15.9 (~2.6x fewer) but only recovers 34.0% of the
  true k=5 nearest neighbors. Manual inspection (not asserted in the
  test, just checked while calibrating `leaf_size`) confirmed *why*: with
  `leaf_size=20` and ~60 points per cluster, each cluster spans 2-3
  sibling leaves, and defeatist descent reliably lands in the correct
  cluster but not always the single closest leaf within it — the
  approximate results are real near-neighbors from the right cluster,
  just not the literal 5 closest points. `leaf_size` directly trades
  recall for node visits (measured while calibrating this test: 10 ->
  21.5% recall/7.3 visits, 20 -> 34.0%/6.2, 30 -> 45.0%/5.6) — a real,
  measured version of the classic ANN speed/recall knob.
- The classic k bias-variance tradeoff, measured: with 25% of training
  labels flipped near the decision boundary, k=1 gets 75.8% held-out
  accuracy (it just copies whichever single point happens to be
  nearest, noisy or not) while k=15 gets 100% (majority vote across 15
  neighbors out-votes the noisy minority) — the same shape of result as
  `decision_tree`'s depth sweep and `svm`'s C sweep, this time for k-NN's
  own hyperparameter.
- KDTree and BallTree's exact mode produce byte-identical classification
  accuracy on the same task (both 100% on two well-separated blobs) —
  expected, since both implement the same exact-neighbor semantics over
  different tree structures; a genuine sanity check that they agree
  rather than an assumption.

## Hardware notes
None for the algorithm itself, correctness, or parameter-effect results
above. Real dataset comparison against sklearn on OpenML CC-18 needs the
deferred Python + `scikit-learn` + dataset-fetch install (tracked in
project memory alongside the JAX/Java/LightGBM deferrals).

---

## cosine_ann (PLAN.md Phase 13 step 2)

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

### What this measures

"Extend ml/knn's ball tree to cosine similarity (currently
Euclidean-only)" — the baseline retrieval index `rag/`'s embedding vectors
(Phase 13 step 1, always L2-normalized) get searched against.

### Design

`CosineBallTree` is a thin wrapper around `BallTree` above, not a modified
copy of it — composition, matching Phase 13's own "RAG is a composition of
existing capabilities" framing. The trick (standard, but verified directly
here rather than just asserted): for **unit** vectors,
`||a-b||^2 = 2 - 2*cos(a,b)`, so ranking by squared Euclidean distance
after L2-normalizing every vector produces *exactly* the same order as
ranking by cosine similarity. `CosineBallTree` normalizes every point once
at construction and the query once per call, then delegates to
`BallTree::query_knn` completely unchanged — no new pruning logic, no new
correctness argument, just a change of coordinates. `query_knn` converts
the underlying (normalized-space) squared distance back to a cosine
similarity value for the caller (`1 - dist_sq/2`).

### Results (captured 2026-07-29, Apple clang 14 / `-std=c++2b`, this Mac)

```
PASS  cosine_similarity of parallel (same-direction) vectors is 1.0
PASS  cosine_similarity of orthogonal vectors is 0.0
PASS  cosine_similarity of opposite-direction vectors is -1.0
  CosineBallTree: 0/20 queries mismatched brute-force cosine ranking; 0/100 similarity values off
PASS  CosineBallTree's exact mode finds the same top-k as brute-force cosine similarity
PASS  CosineBallTree reports the correct cosine similarity value, not just the correct order
  CosineBallTree defeatist: recall=0.600 vs exact, avg nodes visited exact=11.2 approx=6.2
PASS  defeatist cosine search actually misses some true neighbors (a genuine approximation)
PASS  defeatist cosine search still finds a nontrivial fraction of true neighbors
PASS  defeatist cosine search visits fewer nodes than exact backtracking search
PASS
```

### Findings

- The monotonic-transform equivalence is exact, not approximate: 0/20
  queries had a different top-5 SET than brute-force cosine ranking, and
  0/100 reported similarity VALUES were off by more than 1e-4 from direct
  cosine similarity on the original (unnormalized) vectors — confirms
  `CosineBallTree` isn't just "close enough," it's mathematically the same
  ranking, reusing `BallTree`'s already-proven exactness argument for free.
- The approximate (defeatist) mode's recall/speed tradeoff carries over
  from plain `BallTree` largely unchanged in shape: 6.2 avg nodes visited
  vs. 11.2 for exact (~1.8x fewer) at 60.0% recall on 5 well-separated
  12-D clusters — same qualitative "real tradeoff, not free" finding
  `knn/README.md`'s own defeatist-mode section documents, now confirmed to
  hold under the cosine metric too, not just Euclidean.
- This is the retrieval index `indexing_pipeline` (step 4) and
  `recall_eval` (step 6) build on directly — no separate ranking logic to
  keep in sync with `rag/embedding_model`'s L2-normalized output.

### Hardware notes
None — pure CPU, same as `BallTree` itself.
