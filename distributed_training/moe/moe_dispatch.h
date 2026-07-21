#pragma once

// Mixture-of-Experts token dispatch: PLAN.md step 16. One expert per rank
// (E == world_size — the simplest faithful MoE placement), top-1 routing
// (PLAN says "top-k"; top-1 is the k=1 case and keeps this tractable while
// exercising the real mechanism: dispatch, not the routing policy).
//
// The real, previously-unbuilt primitive this step needs is all-to-all
// with VARIABLE per-pair message sizes: different ranks route different
// numbers of tokens to each expert, and networking/common::Channel's
// send/recv contract requires both sides to already know the exact byte
// count (see channel.h — there is no framing header on the wire, by
// design, for the FIXED-size collectives steps 11-15 needed). MoE dispatch
// doesn't have a fixed size, so this is a genuine three-phase protocol:
// exchange counts (fixed size: one int per pair), dispatch each token's
// data to its assigned expert now that both sides know the size, then
// COMBINE the expert outputs back to their origin. Combine needs no
// separate index bookkeeping: both sides already know the exact group
// sizes from the dispatch phase, and TCP's in-order delivery means row k
// of a reply group is unambiguously the k-th token that group's sender
// originally sent — the same trick real alltoallv implementations rely on.
//
// Forward-only: this validates routing + dispatch + expert compute +
// combine against a single-process reference. Training (backward through
// the router) is out of scope — top-1's argmax is not differentiable in
// the naive sense (real systems use a straight-through estimator or treat
// only the gate WEIGHT, not the discrete choice, as differentiable, which
// is a separate design decision from what this step validates).

#include <cstddef>
#include <functional>
#include <vector>

#include "channel.h"
#include "matrix.h"

namespace distributed_training {

// Top-1 routing: returns, for each row (token) of `gate_logits` [num_tokens
// x num_experts], the argmax expert index.
inline std::vector<int> route_top1(const Matrix &gate_logits) {
  std::vector<int> assignment(static_cast<size_t>(gate_logits.rows()));
  for (int i = 0; i < gate_logits.rows(); ++i) {
    int best = 0;
    float best_val = gate_logits(i, 0);
    for (int j = 1; j < gate_logits.cols(); ++j) {
      if (gate_logits(i, j) > best_val) { best_val = gate_logits(i, j); best = j; }
    }
    assignment[static_cast<size_t>(i)] = best;
  }
  return assignment;
}

// Runs one MoE forward pass: routes local_x's rows (this rank's shard of
// the global batch) via `dest_rank` (one destination rank == expert id per
// row, from route_top1 over this rank's gate logits), dispatches each row
// to its assigned rank, applies `expert_fn` (this rank's own local expert)
// to every row it receives (from any rank, including its own — handled
// locally with no network round-trip), sends results back to each row's
// origin, and returns the combined output in the SAME row order as
// local_x. Every rank must call this together — it is a collective, like
// the fixed-size ones in collectives.h.
inline Matrix moe_forward(const Matrix &local_x, const std::vector<int> &dest_rank,
                           const std::function<Matrix(const Matrix &)> &expert_fn, netcommon::Channel &channel) {
  int world = channel.world_size();
  int me = channel.rank();
  int hidden = local_x.cols();

  std::vector<std::vector<int>> buckets(static_cast<size_t>(world)); // local row indices per destination
  for (size_t i = 0; i < dest_rank.size(); ++i) buckets[static_cast<size_t>(dest_rank[i])].push_back(static_cast<int>(i));

  std::vector<int> send_counts(static_cast<size_t>(world)), recv_counts(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) send_counts[static_cast<size_t>(r)] = static_cast<int>(buckets[static_cast<size_t>(r)].size());

  // Phase 1: exchange counts. Deadlock-safe pairwise ordering (lower rank
  // sends first) — safe per-pair since Channel is a full mesh, one socket
  // per pair (see channel.h); different pairs never interfere.
  for (int r = 0; r < world; ++r) {
    if (r == me) { recv_counts[static_cast<size_t>(r)] = send_counts[static_cast<size_t>(r)]; continue; }
    if (me < r) {
      channel.send(r, &send_counts[static_cast<size_t>(r)], sizeof(int));
      channel.recv(r, &recv_counts[static_cast<size_t>(r)], sizeof(int));
    } else {
      channel.recv(r, &recv_counts[static_cast<size_t>(r)], sizeof(int));
      channel.send(r, &send_counts[static_cast<size_t>(r)], sizeof(int));
    }
  }

  // Phase 2: dispatch — exchange the actual token data, grouped by source
  // rank IN RANK ORDER (0..world-1) so the combine phase below can rely on
  // that same grouping without extra bookkeeping.
  int total_recv = 0;
  for (int c : recv_counts) total_recv += c;
  Matrix received(total_recv, hidden);
  int offset = 0;
  for (int r = 0; r < world; ++r) {
    if (r == me) {
      for (int local_idx : buckets[static_cast<size_t>(r)]) {
        for (int j = 0; j < hidden; ++j) received(offset, j) = local_x(local_idx, j);
        ++offset;
      }
      continue;
    }
    std::vector<float> send_buf(static_cast<size_t>(send_counts[static_cast<size_t>(r)]) * static_cast<size_t>(hidden));
    for (size_t k = 0; k < buckets[static_cast<size_t>(r)].size(); ++k) {
      int local_idx = buckets[static_cast<size_t>(r)][k];
      for (int j = 0; j < hidden; ++j) send_buf[k * static_cast<size_t>(hidden) + static_cast<size_t>(j)] = local_x(local_idx, j);
    }
    std::vector<float> recv_buf(static_cast<size_t>(recv_counts[static_cast<size_t>(r)]) * static_cast<size_t>(hidden));
    if (me < r) {
      if (!send_buf.empty()) channel.send(r, send_buf.data(), send_buf.size() * sizeof(float));
      if (!recv_buf.empty()) channel.recv(r, recv_buf.data(), recv_buf.size() * sizeof(float));
    } else {
      if (!recv_buf.empty()) channel.recv(r, recv_buf.data(), recv_buf.size() * sizeof(float));
      if (!send_buf.empty()) channel.send(r, send_buf.data(), send_buf.size() * sizeof(float));
    }
    for (int k = 0; k < recv_counts[static_cast<size_t>(r)]; ++k) {
      for (int j = 0; j < hidden; ++j) received(offset, j) = recv_buf[static_cast<size_t>(k) * static_cast<size_t>(hidden) + static_cast<size_t>(j)];
      ++offset;
    }
  }

  // Local expert compute on everything this rank received, regardless of origin.
  Matrix processed = expert_fn(received);
  int hidden_out = processed.cols();

  // Phase 3: combine — send each group of results back to its origin, in
  // the SAME per-rank grouping/order used in phase 2, so origin rank r
  // receiving `send_counts[me]` rows back (a size it already knows from
  // phase 1) can assign row k directly to buckets[r][k] with no further
  // metadata.
  Matrix output(local_x.rows(), hidden_out);
  offset = 0;
  for (int r = 0; r < world; ++r) {
    if (r == me) {
      for (int local_idx : buckets[static_cast<size_t>(r)]) {
        for (int j = 0; j < hidden_out; ++j) output(local_idx, j) = processed(offset, j);
        ++offset;
      }
      continue;
    }
    std::vector<float> reply_send(static_cast<size_t>(recv_counts[static_cast<size_t>(r)]) * static_cast<size_t>(hidden_out));
    for (int k = 0; k < recv_counts[static_cast<size_t>(r)]; ++k) {
      for (int j = 0; j < hidden_out; ++j) reply_send[static_cast<size_t>(k) * static_cast<size_t>(hidden_out) + static_cast<size_t>(j)] = processed(offset + k, j);
    }
    offset += recv_counts[static_cast<size_t>(r)];

    std::vector<float> reply_recv(static_cast<size_t>(send_counts[static_cast<size_t>(r)]) * static_cast<size_t>(hidden_out));
    if (me < r) {
      if (!reply_send.empty()) channel.send(r, reply_send.data(), reply_send.size() * sizeof(float));
      if (!reply_recv.empty()) channel.recv(r, reply_recv.data(), reply_recv.size() * sizeof(float));
    } else {
      if (!reply_recv.empty()) channel.recv(r, reply_recv.data(), reply_recv.size() * sizeof(float));
      if (!reply_send.empty()) channel.send(r, reply_send.data(), reply_send.size() * sizeof(float));
    }
    for (size_t k = 0; k < buckets[static_cast<size_t>(r)].size(); ++k) {
      int local_idx = buckets[static_cast<size_t>(r)][k];
      for (int j = 0; j < hidden_out; ++j) output(local_idx, j) = reply_recv[k * static_cast<size_t>(hidden_out) + static_cast<size_t>(j)];
    }
  }

  return output;
}

} // namespace distributed_training
