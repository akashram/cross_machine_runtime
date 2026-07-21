# MoE / Expert Parallelism

**Status: code-complete AND locally run — portable, builds on
autograd/matrix.h and networking/common (raw Channel send/recv).**

## What this measures

PLAN.md Phase 6 step 16: top-k routing, expert dispatch via all-to-all,
expert forward pass, combine results, expert utilization distribution.
Validated numerically against a single-process reference; top-1 routing
(the k=1 case — see Design for why), forward-only (see Design for why
training is out of scope here).

## Design

One expert per rank, top-1 routing (`route_top1`). The real, previously-
unbuilt primitive: `networking/collectives` only has FIXED-size
collectives (every rank sends/receives the same count) — MoE dispatch is
inherently variable-size (different ranks route different numbers of
tokens to each expert), and `Channel`'s send/recv contract requires both
sides to already know the exact byte count (no wire framing, by design —
see `channel.h`). `moe_forward` (`moe_dispatch.h`) is a genuine three-phase
protocol: exchange counts (fixed size), dispatch tokens now that sizes are
known, then COMBINE expert outputs back to origin — combine needs no
extra index bookkeeping, since both sides already know the group sizes
from phase 1, and TCP's in-order delivery means row k of a reply is
unambiguously the k-th token that peer sent.

**Scoping, stated plainly**: forward-only. Backward through the router is
out of scope — top-1's argmax is not differentiable in the naive sense
(real systems use a straight-through estimator or make only the gate
WEIGHT differentiable, a separate design decision from what this step
validates: dispatch/combine correctness).

## Sanity-run output (Mac, loopback, 2026-07-21)

4 ranks, 4 experts, 20 tokens (5/rank), untrained random gate weights:

```
MoE dispatch/combine round-trip: max abs diff vs single-process reference = 0.000000: PASS
expert load (tokens routed to each expert, out of 20 total):
  expert 0: 3 tokens (15.0%)
  expert 1: 8 tokens (40.0%)
  expert 2: 4 tokens (20.0%)
  expert 3: 5 tokens (25.0%)
PASS
```

Exact bit-for-bit match against the single-process reference — every token
survives the dispatch/combine round-trip and lands on the correct expert.
The expert load is visibly imbalanced (15%-40%, vs. an even 25% each) —
expected with an untrained random gate, and a real illustration of why
production MoE systems need a load-balancing auxiliary loss (not built
here — this step validates the mechanism, not training-time balance).

## Results
TODO: run on GPU hardware — the numbers that matter are all-to-all
communication overhead at real batch/hidden-dim scale and real expert
load distribution after actual training (which requires the router's
backward path this step deliberately left out).

| Experts | World size | Tokens | All-to-all overhead | Expert load (post-training) |
|---------|--------------|----------|------------------------|----------------------------------|
| TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Overhead/load-balance validation (Results table): multi-GPU instance.
