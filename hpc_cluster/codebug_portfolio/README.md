# codebug_portfolio -- HW/SW co-debug case studies from this repo's own real bugs

**Status: written portfolio piece, no new code.** PLAN.md Phase 22 step
8: consolidate this repo's own real cross-boundary debugging stories as
direct evidence for "debugging across software/OS/hardware boundaries" --
honest consolidation of what already happened, not new material invented
for this step.

## Case 1: `networking/raft` -- a real SIGSEGV from a software/OS boundary (thread lifetime vs. socket lifetime)

**The bug class**: a classic software/OS co-debug problem -- a C++
object's lifetime (`RaftNode`, destructed by `unique_ptr` teardown) and
an OS thread's lifetime (a detached receiver thread, still running
inside `Channel::recv()` or deep inside `handleRequestVote`/
`handleAppendEntries`) were assumed to be safely decoupled and weren't.

**The debugging path, not just the fix**: the ORIGINAL design joined
every receiver thread on `stop()` -- correct for stopping a whole
cluster, but it deadlocked the leader-failover test (a "crashed" node's
still-running peers never send it a shutdown frame back to unblock the
join). The FIRST fix -- detaching the threads instead, reasoning "each
one is blocked forever in `recv()`, so it's harmless to leave running" --
looked reasonable on inspection and was wrong: a receiver thread isn't
always blocked in `recv()`, it can be anywhere inside a handler, actively
touching `this`. This only reproduced reliably under real OS scheduling
pressure (6 busy-loop processes on a 2-core Mac, `raft_test` run in a
loop) -- a genuine hardware/OS-timing-dependent bug that inspection alone
would not have found, and that DID produce a real, captured crash report:
`SIGSEGV`/`EXC_BAD_ACCESS` at address `0x0`, faulting thread inside
`recvFrame -> Channel::recv` on the now-dangling `channel_` reference,
called from a detached thread whose owning `RaftNode` had already been
destructed. 0/40 clean runs under contention before diagnosis.

**The real fix, and what it exposed**: `Channel::shutdownPeer(peer)`
(`shutdown(fd, SHUT_RD)` on the underlying socket) forcibly unblocks
*this node's own* `recv()` call with zero cooperation needed from the
peer -- `stop()` now calls this for every peer, THEN genuinely `t.join()`s
every receiver thread, so by the time `stop()` returns every thread the
`RaftNode` owns has actually exited. Fixing the socket-shutdown path
exposed two more real, PRE-EXISTING bugs one layer down in the socket
transport itself: (a) a peer sending to a half-shut-down socket can get
`SIGPIPE`, whose default disposition kills the whole process (not just
the failing call) -- fixed by ignoring `SIGPIPE` process-wide, the
standard portable fix; (b) even with `SIGPIPE` suppressed, a failed
`send()` threw an uncaught `std::runtime_error`, calling
`std::terminate()` -- captured: `libc++abi: terminating due to uncaught
exception ... Connection reset by peer`. Fixed by making `sendFrame`
best-effort (a dropped RPC to an unreachable peer is a NORMAL condition
in a distributed system, not a fatal one -- Raft's own periodic
heartbeats and repeated elections exist specifically to tolerate this).

**Verification, not just a fix and move on**: 100/100 clean `raft_test`
runs under the SAME artificial CPU contention that reliably crashed both
the original detach-based code and the incomplete first fix; also
TSan-clean under contention. See `networking/raft/README.md` for the
full account.

## Case 2: `fpga_engine/cocotb` -- a real one-cycle timing bug, caught by RTL simulation, not by reading the FSM

**The bug class**: a hardware-timing bug in synchronous digital logic --
the kind of bug that "looks correct on inspection" precisely because the
FSM's state names and structure are right; only the exact clock-cycle
alignment between a requester's read strobe and a synchronous memory's
output timing was wrong.

**The bug**: the DMA controller's first version had only ONE wait state
between asserting `mem_rden` and sampling `mem_rdata`. The actual timing
relationship: `mem_rden`/`mem_addr` become stable starting at the edge
that asserts them (E0); a registered memory samples them and produces
valid `mem_rdata` starting at the FOLLOWING edge (E1, stable through
`[E1, E2)`) -- so the requester can only safely sample at E2, not E1. The
single-wait-state version sampled at E1, one cycle too early.

**How it was caught**: not by re-reading the FSM (which "looked" right),
but by running `test_dma_copy` against cocotb's posedge-synchronous
memory model -- a real RTL simulation exposed it immediately as a
one-word LAG in every copied value, a completely unambiguous signature
once observed:
```
AssertionError: copy mismatch: src=[40960, 40961, 40962, 40963, 40964, 40965, 40966, 40967]
dst=[0, 40960, 40961, 40962, 40963, 40964, 40965, 40966]
```

**The fix and verification**: split the single wait state into two
(`S_READ_WAIT1`, `S_READ_WAIT2` in `dma_controller.v`), giving a full
extra cycle of margin before sampling `mem_rdata`. Re-running the test
confirmed the fix. The same RTL was independently, formally proven
correct one step later (`fpga_engine/symbiyosys`'s k-induction proof that
`mem_rden`/`mem_wren` are never simultaneously asserted) -- two
DIFFERENT verification methods (dynamic simulation, static formal proof)
against the SAME post-fix RTL, neither one a substitute for the other:
simulation caught a TIMING bug formal verification's specific properties
didn't target, and the formal proof later covered every reachable state
exhaustively, which simulation's finite test vectors cannot. See
`fpga_engine/cocotb/README.md` for the full account.

## Case 3: `ml/pca` -- a real numerical-precision bug, ill-conditioning as the actual root cause

**The bug class**: not a hardware-timing or lifetime bug like cases 1-2,
but the software/numerics co-debug problem of a fixed-precision
floating-point type silently losing enough relative precision under real
ill-conditioned data to produce a physically impossible output.

**The bug**: sweeping PCA's `n_components` against the real `wdbc`
OpenML dataset (30 raw, unstandardized features spanning `[0.05, 2500]`
-- a condition number in the thousands) produced `cum_var_ratio` values
ABOVE 1.0 (1.613 at `n_components=2`, rising to 1.989 at
`n_components=30`) -- a ratio that must sum to at most 1 by definition,
so this was unambiguously wrong, not just suspicious.

**Root cause, traced (not guessed)**: `pca.cpp`'s randomized-SVD linear
algebra (`matmul`, `orthonormalize_columns`, `jacobi_eigen`) accumulated
in `float` (~7 significant digits). The power-iteration step
(`Y = A(A^T Y)`, repeated squaring toward the dominant singular
direction) lost enough relative precision in the WEAKER singular
directions -- specifically under `wdbc`'s scale of ill-conditioning, a
~50,000x ratio of feature standard deviations -- that their reported
variance came out inflated. `pca_test.cpp`'s own existing unit tests
never caught this because they use synthetic, WELL-scaled data (max
feature-stddev ratio of only 10x) -- a real illustration of why a unit
test suite passing is not the same claim as "correct at the scale/
conditioning real data actually has."

**The fix and verification**: switched the internal linear algebra to
`double` (~15-16 significant digits) end to end -- mean/covariance
accumulation, the Gaussian sketch, QR, Jacobi eigendecomposition --
converting back to `float` only at the public API boundary
(`components_`, `singular_values_`), so external callers see no type
change. Re-ran `pca_test`: zero regression on every existing check, and
the `wdbc` sweep now rises monotonically toward 1.0 as it physically
must. See `ml/hyperparam_sensitivity/README.md` for the full account
(the bug surfaced there, in the real cross-dataset hyperparameter sweep,
not in `pca`'s own unit tests).

## What these three cases have in common, and what's different

All three are REAL bugs, caught by actually RUNNING code under
conditions that stressed a specific real boundary (OS scheduling
pressure for case 1, exact clock-cycle timing for case 2, real
ill-conditioned data for case 3) -- none was found by code review alone,
and in each case the bug "looked right" on inspection precisely because
the surrounding structure (the FSM, the reasoning about thread safety,
the linear-algebra formulas) was correct; only one specific boundary
condition was wrong. They span three genuinely different boundary types
a co-debug role has to move between fluently: software/OS (thread
lifetime vs. socket lifetime, case 1), software/hardware-timing (RTL
clock-cycle alignment, case 2), and software/numerics (floating-point
precision vs. real data conditioning, case 3) -- the third is the one
most often left off a "hardware co-debug" portfolio, but is exactly as
real a cross-boundary bug as the other two, and arguably the one closest
to what an ML/HPC-infrastructure debugging role encounters most often in
practice (a training run producing subtly wrong numbers, not a crash).

## Hardware notes
None -- this step is pure written consolidation of three already-real,
already-fixed, already-documented bugs. No new measurement.
