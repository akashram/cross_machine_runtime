#include "srd_selector.h"

namespace {
// Placeholder threshold — TODO: replace with the real crossover measured
// once EFA hardware is available (README's Results table). Ring
// all-reduce only ever has 2 peers (its neighbors) regardless of world
// size, so it stays RC-eligible on peer count alone even at large scale;
// it's patterns like all-to-all MoE dispatch or the gRPC control plane's
// fan-out that push num_peers up and tip the decision toward SRD.
constexpr int kPeerCountThreshold = 8;
} // namespace

EfaTransport select_transport(const WorkloadProfile &profile) {
  if (profile.ordering_required && profile.num_peers < kPeerCountThreshold)
    return EfaTransport::RC;
  if (profile.num_peers >= kPeerCountThreshold)
    return EfaTransport::SRD; // QP-per-peer cost of RC dominates
  return profile.ordering_required ? EfaTransport::RC : EfaTransport::SRD;
}
