#pragma once

// PLAN.md Phase 6 step 6: reverse-mode tape-based autograd, chosen over
// "explicit interface to PyTorch autograd" (PLAN.md's alternative) because
// there is no PyTorch/libtorch dependency anywhere else in this project
// (see CLAUDE.md's tooling decisions — this codebase writes its own
// primitives at every layer, including the coroutine engine and lock-free
// structures in foundation/), and because steps 7-25 need to shard and
// manipulate gradients directly (ZeRO, tensor/pipeline parallelism), which
// is far more natural against an autograd engine this project owns than
// against an opaque external one.
//
// Design: each Tensor holds a shared_ptr<Node> (the tape). Node stores a
// value Matrix, a grad Matrix of the same shape (accumulated, not
// overwritten — a Node used by two consumers must sum both contributions,
// the standard multivariate chain rule for a DAG rather than a tree), its
// parent Nodes, and a backward_fn closure that — given this Node's
// (already-fully-accumulated) grad — adds this op's contribution into each
// parent's grad. backward() topologically sorts the DAG from the output
// and calls backward_fn on each node in reverse-topological order, which
// guarantees every consumer of a Node has already contributed to its grad
// before that Node computes contributions to ITS parents.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "matrix.h"

namespace distributed_training {

struct Node {
  Matrix value;
  Matrix grad;
  std::vector<std::shared_ptr<Node>> parents;
  std::function<void()> backward_fn;

  explicit Node(Matrix v) : value(std::move(v)), grad(value.rows(), value.cols()) {}
};

class Tensor {
public:
  Tensor() = default;
  explicit Tensor(Matrix value) : node_(std::make_shared<Node>(std::move(value))) {}
  explicit Tensor(std::shared_ptr<Node> node) : node_(std::move(node)) {}

  const Matrix &value() const { return node_->value; }
  const Matrix &grad() const { return node_->grad; }
  // Both mutate through the shared_ptr indirection (shallow const, like a
  // raw pointer) so they are callable on a `const Tensor&` — the Tensor
  // handle itself is not modified, only the Node it points to, which is
  // exactly what an optimizer or zero_grad() needs to do without requiring
  // every parameter list to be threaded through as non-const.
  Matrix &mutable_value() const { return node_->value; } // for in-place optimizer updates on leaf parameters
  void zero_grad() const { node_->grad = Matrix(node_->value.rows(), node_->value.cols()); }

  Node *raw() const { return node_.get(); }
  const std::shared_ptr<Node> &node() const { return node_; }

  int rows() const { return node_->value.rows(); }
  int cols() const { return node_->value.cols(); }

  // Runs backward from this Tensor (must be scalar, 1x1 — a loss) with
  // seed gradient 1.0, accumulating into every ancestor's grad.
  void backward() const {
    assert(rows() == 1 && cols() == 1);
    node_->grad = Matrix(1, 1);
    node_->grad(0, 0) = 1.0f;
    run_backward();
  }

  // Same, but with an explicit seed (for gradient-checking utilities that
  // need to backward from a non-scalar node with a specific direction).
  void backward_with_seed(const Matrix &seed) const {
    node_->grad = seed;
    run_backward();
  }

private:
  void run_backward() const {
    std::vector<Node *> order;
    std::unordered_set<Node *> visited;
    topo_sort(node_.get(), visited, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      if ((*it)->backward_fn) (*it)->backward_fn();
    }
  }

  static void topo_sort(Node *n, std::unordered_set<Node *> &visited, std::vector<Node *> &order) {
    if (visited.count(n)) return;
    visited.insert(n);
    for (auto &p : n->parents) topo_sort(p.get(), visited, order);
    order.push_back(n);
  }

  std::shared_ptr<Node> node_;
};

// ---------------------------------------------------------------------
// Ops. Each builds a new Node whose backward_fn implements that op's
// contribution to its parents' grad given ITS OWN (fully accumulated) grad.
// ---------------------------------------------------------------------

inline Tensor matmul(const Tensor &a, const Tensor &b) {
  auto out_node = std::make_shared<Node>(a.value().matmul(b.value()));
  out_node->parents = {a.node(), b.node()};
  Node *an = a.raw();
  Node *bn = b.raw();
  Node *on = out_node.get();
  out_node->backward_fn = [an, bn, on]() {
    an->grad.add_inplace(on->grad.matmul(bn->value.transpose()));
    bn->grad.add_inplace(an->value.transpose().matmul(on->grad));
  };
  return Tensor(out_node);
}

inline Tensor add(const Tensor &a, const Tensor &b) {
  auto out_node = std::make_shared<Node>(a.value() + b.value());
  out_node->parents = {a.node(), b.node()};
  Node *an = a.raw();
  Node *bn = b.raw();
  Node *on = out_node.get();
  out_node->backward_fn = [an, bn, on]() {
    an->grad.add_inplace(on->grad);
    bn->grad.add_inplace(on->grad);
  };
  return Tensor(out_node);
}

// Broadcast-adds a [1 x cols] bias to every row of a [rows x cols] input.
inline Tensor add_bias(const Tensor &a, const Tensor &bias) {
  auto out_node = std::make_shared<Node>(a.value().add_row_broadcast(bias.value()));
  out_node->parents = {a.node(), bias.node()};
  Node *an = a.raw();
  Node *bn = bias.raw();
  Node *on = out_node.get();
  out_node->backward_fn = [an, bn, on]() {
    an->grad.add_inplace(on->grad);
    bn->grad.add_inplace(on->grad.sum_rows());
  };
  return Tensor(out_node);
}

inline Tensor relu(const Tensor &a) {
  auto out_node = std::make_shared<Node>(a.value().apply([](float v) { return v > 0.0f ? v : 0.0f; }));
  out_node->parents = {a.node()};
  Node *an = a.raw();
  Node *on = out_node.get();
  out_node->backward_fn = [an, on]() {
    Matrix mask = an->value.apply([](float v) { return v > 0.0f ? 1.0f : 0.0f; });
    an->grad.add_inplace(on->grad.elementwise_mul(mask));
  };
  return Tensor(out_node);
}

// Softmax + cross-entropy, computed together for numerical stability (the
// standard trick: the combined backward is simply softmax(logits) - onehot,
// which avoids ever dividing by a possibly-tiny softmax denominator in the
// backward pass the way separate softmax-then-log-then-nll ops would).
// `labels[i]` is the true class index for row i. Returns a 1x1 scalar mean
// loss over the batch.
inline Tensor softmax_cross_entropy(const Tensor &logits, const std::vector<int> &labels) {
  const Matrix &z = logits.value();
  int batch = z.rows();
  int classes = z.cols();
  Matrix probs(batch, classes);
  float total_loss = 0.0f;
  for (int i = 0; i < batch; ++i) {
    float max_logit = z(i, 0);
    for (int j = 1; j < classes; ++j) max_logit = std::max(max_logit, z(i, j));
    float denom = 0.0f;
    for (int j = 0; j < classes; ++j) denom += std::exp(z(i, j) - max_logit);
    for (int j = 0; j < classes; ++j) probs(i, j) = std::exp(z(i, j) - max_logit) / denom;
    total_loss += -std::log(std::max(probs(i, labels[static_cast<size_t>(i)]), 1e-9f));
  }

  Matrix loss_val(1, 1);
  loss_val(0, 0) = total_loss / static_cast<float>(batch);
  auto out_node = std::make_shared<Node>(loss_val);
  out_node->parents = {logits.node()};
  Node *ln = logits.raw();
  Node *on = out_node.get();
  out_node->backward_fn = [ln, on, probs, labels, batch, classes]() {
    float seed = on->grad(0, 0); // usually 1.0, from Tensor::backward()
    Matrix dlogits(batch, classes);
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < classes; ++j) {
        float onehot = (j == labels[static_cast<size_t>(i)]) ? 1.0f : 0.0f;
        dlogits(i, j) = (probs(i, j) - onehot) / static_cast<float>(batch) * seed;
      }
    }
    ln->grad.add_inplace(dlogits);
  };
  return Tensor(out_node);
}

} // namespace distributed_training
