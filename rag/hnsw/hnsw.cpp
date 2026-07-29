#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>

namespace {
bool by_dist_less(const NeighborResult &a, const NeighborResult &b) { return a.dist_sq < b.dist_sq; }
} // namespace

HNSW::HNSW(const Features &points) : HNSW(points, Params{}) {}

HNSW::HNSW(const Features &points, Params params) : points_(points), params_(params), rng_(params.seed) {
  mL_ = 1.0f / std::log(static_cast<float>(std::max(params_.M, 2)));
  nodes_.resize(points_.size());
  for (std::size_t i = 0; i < points_.size(); ++i) insert(static_cast<int>(i));
}

int HNSW::random_level() {
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  float r = std::max(uni(rng_), 1e-12f);
  return static_cast<int>(std::floor(-std::log(r) * mL_));
}

std::vector<int> HNSW::select_neighbors_simple(std::vector<NeighborResult> candidates, int m) {
  std::sort(candidates.begin(), candidates.end(), by_dist_less);
  if (static_cast<int>(candidates.size()) > m) candidates.resize(static_cast<std::size_t>(m));
  std::vector<int> out;
  out.reserve(candidates.size());
  for (const auto &c : candidates) out.push_back(c.index);
  return out;
}

std::vector<NeighborResult> HNSW::search_layer(const std::vector<float> &query, const std::vector<int> &entry_points,
                                                 int ef, int layer) const {
  auto min_cmp = [](const NeighborResult &a, const NeighborResult &b) { return a.dist_sq > b.dist_sq; };
  auto max_cmp = [](const NeighborResult &a, const NeighborResult &b) { return a.dist_sq < b.dist_sq; };
  std::priority_queue<NeighborResult, std::vector<NeighborResult>, decltype(min_cmp)> candidates(min_cmp);
  std::priority_queue<NeighborResult, std::vector<NeighborResult>, decltype(max_cmp)> found(max_cmp);
  std::set<int> visited;

  for (int ep : entry_points) {
    float d = squared_distance(points_[static_cast<std::size_t>(ep)], query);
    ++last_distance_evals_;
    candidates.push({ep, d});
    found.push({ep, d});
    visited.insert(ep);
  }

  while (!candidates.empty()) {
    NeighborResult c = candidates.top();
    candidates.pop();
    if (static_cast<int>(found.size()) >= ef && c.dist_sq > found.top().dist_sq) break;

    for (int neighbor : nodes_[static_cast<std::size_t>(c.index)].layers[static_cast<std::size_t>(layer)]) {
      if (visited.count(neighbor)) continue;
      visited.insert(neighbor);
      float d = squared_distance(points_[static_cast<std::size_t>(neighbor)], query);
      ++last_distance_evals_;
      if (static_cast<int>(found.size()) < ef || d < found.top().dist_sq) {
        candidates.push({neighbor, d});
        found.push({neighbor, d});
        if (static_cast<int>(found.size()) > ef) found.pop();
      }
    }
  }

  std::vector<NeighborResult> result;
  result.reserve(found.size());
  while (!found.empty()) { result.push_back(found.top()); found.pop(); }
  std::sort(result.begin(), result.end(), by_dist_less);
  return result;
}

void HNSW::insert(int idx) {
  int level = random_level();

  if (entry_point_ == -1) {
    nodes_[static_cast<std::size_t>(idx)].layers.resize(static_cast<std::size_t>(level + 1));
    entry_point_ = idx;
    max_layer_ = level;
    return;
  }

  std::vector<int> ep{entry_point_};
  for (int lc = max_layer_; lc > level; --lc) {
    auto found = search_layer(points_[static_cast<std::size_t>(idx)], ep, 1, lc);
    ep = {found[0].index};
  }

  nodes_[static_cast<std::size_t>(idx)].layers.resize(static_cast<std::size_t>(level + 1));
  for (int lc = std::min(level, max_layer_); lc >= 0; --lc) {
    auto candidates = search_layer(points_[static_cast<std::size_t>(idx)], ep, params_.ef_construction, lc);
    int m = (lc == 0) ? params_.M * 2 : params_.M;
    auto selected = select_neighbors_simple(candidates, m);
    nodes_[static_cast<std::size_t>(idx)].layers[static_cast<std::size_t>(lc)] = selected;

    for (int nb : selected) {
      auto &nb_layer = nodes_[static_cast<std::size_t>(nb)].layers[static_cast<std::size_t>(lc)];
      nb_layer.push_back(idx);
      if (static_cast<int>(nb_layer.size()) > m) {
        std::vector<NeighborResult> nb_candidates;
        nb_candidates.reserve(nb_layer.size());
        for (int other : nb_layer)
          nb_candidates.push_back({other, squared_distance(points_[static_cast<std::size_t>(nb)],
                                                             points_[static_cast<std::size_t>(other)])});
        nb_layer = select_neighbors_simple(nb_candidates, m);
      }
    }

    ep.clear();
    for (const auto &c : candidates) ep.push_back(c.index);
  }

  if (level > max_layer_) {
    max_layer_ = level;
    entry_point_ = idx;
  }
}

std::vector<NeighborResult> HNSW::query_knn(const std::vector<float> &query, int k) const {
  last_distance_evals_ = 0;
  if (entry_point_ == -1) return {};

  std::vector<int> ep{entry_point_};
  for (int lc = max_layer_; lc > 0; --lc) {
    auto found = search_layer(query, ep, 1, lc);
    ep = {found[0].index};
  }

  auto found = search_layer(query, ep, std::max(params_.ef_search, k), 0);
  if (static_cast<int>(found.size()) > k) found.resize(static_cast<std::size_t>(k));
  return found;
}
