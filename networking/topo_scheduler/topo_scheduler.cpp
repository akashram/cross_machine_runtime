//===- topo_scheduler.cpp - Step 16 implementation ------------------------===//

#include "topo_scheduler.h"

#include <algorithm>
#include <limits>

namespace topo {

int TopologyGraph::addNode(const std::string &name) {
  nodeNames.push_back(name);
  links.emplace_back();
  return static_cast<int>(nodeNames.size()) - 1;
}

void TopologyGraph::addLink(int a, int b, double bandwidth_gbs) {
  links[a].push_back({b, bandwidth_gbs});
  links[b].push_back({a, bandwidth_gbs});
}

namespace {

// All-pairs cost via Floyd-Warshall, cost = sum of 1/bandwidth along the
// path (treats bandwidth like electrical conductance — lower cost is a
// faster path, and costs of chained hops add, matching how transfer time
// over a multi-hop path roughly composes for this placement-level
// approximation).
std::vector<std::vector<double>> allPairsCost(const TopologyGraph &g) {
  size_t n = g.nodeNames.size();
  constexpr double kInf = std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> cost(n, std::vector<double>(n, kInf));
  for (size_t i = 0; i < n; ++i) cost[i][i] = 0.0;
  for (size_t i = 0; i < n; ++i)
    for (const Edge &e : g.links[i])
      cost[i][static_cast<size_t>(e.to)] = std::min(cost[i][static_cast<size_t>(e.to)], 1.0 / e.bandwidth_gbs);

  for (size_t k = 0; k < n; ++k)
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if (cost[i][k] + cost[k][j] < cost[i][j])
          cost[i][j] = cost[i][k] + cost[k][j];
  return cost;
}

} // namespace

std::vector<int> place(const TopologyGraph &topology, const Workload &workload) {
  int numNodes = static_cast<int>(topology.nodeNames.size());
  std::vector<std::vector<double>> cost = allPairsCost(topology);

  struct Pair { int i, j; double volume; };
  std::vector<Pair> pairs;
  for (int i = 0; i < workload.numTasks; ++i)
    for (int j = i + 1; j < workload.numTasks; ++j)
      if (workload.commVolumeGb[i][j] > 0)
        pairs.push_back({i, j, workload.commVolumeGb[i][j]});
  std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) { return a.volume > b.volume; });

  std::vector<int> taskToNode(workload.numTasks, -1);
  std::vector<bool> nodeUsed(numNodes, false);

  auto bestFreeNode = [&](int fixedNode) {
    int best = -1;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int n = 0; n < numNodes; ++n) {
      if (nodeUsed[n]) continue;
      double c = (fixedNode < 0) ? 0.0 : cost[static_cast<size_t>(fixedNode)][static_cast<size_t>(n)];
      if (c < bestCost) { bestCost = c; best = n; }
    }
    return best;
  };

  for (const Pair &p : pairs) {
    bool iPlaced = taskToNode[p.i] >= 0;
    bool jPlaced = taskToNode[p.j] >= 0;
    if (iPlaced && jPlaced) continue;

    if (!iPlaced && !jPlaced) {
      // Neither placed yet: pick the globally cheapest free (a,b) pair —
      // for a fresh pair with no existing anchor, "closest two free
      // nodes" is the natural generalization of bestFreeNode's
      // single-anchor search.
      int bestA = -1, bestB = -1;
      double bestCost = std::numeric_limits<double>::infinity();
      for (int a = 0; a < numNodes; ++a) {
        if (nodeUsed[a]) continue;
        for (int b = 0; b < numNodes; ++b) {
          if (b == a || nodeUsed[b]) continue;
          double c = cost[static_cast<size_t>(a)][static_cast<size_t>(b)];
          if (c < bestCost) { bestCost = c; bestA = a; bestB = b; }
        }
      }
      taskToNode[p.i] = bestA; nodeUsed[bestA] = true;
      taskToNode[p.j] = bestB; nodeUsed[bestB] = true;
    } else if (!iPlaced) {
      int node = bestFreeNode(taskToNode[p.j]);
      taskToNode[p.i] = node; nodeUsed[node] = true;
    } else {
      int node = bestFreeNode(taskToNode[p.i]);
      taskToNode[p.j] = node; nodeUsed[node] = true;
    }
  }

  // Tasks with no communication at all: place arbitrarily on whatever's free.
  for (int t = 0; t < workload.numTasks; ++t) {
    if (taskToNode[t] >= 0) continue;
    int node = bestFreeNode(-1);
    taskToNode[t] = node; nodeUsed[node] = true;
  }

  return taskToNode;
}

} // namespace topo
