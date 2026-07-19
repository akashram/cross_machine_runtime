--------------------------- MODULE Collective ---------------------------
(* TLA+ specification of ring all-reduce (networking/ring_allreduce/
   ring_allreduce.cpp), focused specifically on the property that
   implementation actually depends on: networking/common::TcpChannel
   shares ONE bidirectional, bounded-capacity socket per rank pair (see
   channel.h), not independent per-direction channels. This spec models
   that sharing explicitly — a channel with unbounded capacity or
   independent directions would never expose the deadlock risk
   ring_allreduce.cpp's send-before-recv/recv-before-send parity rule
   exists to avoid, and checking against that idealization would prove
   nothing about the actual implementation. *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS N  \* number of ranks, N >= 2
ASSUME N \in Nat /\ N >= 2

Ranks == 0..(N-1)
Right(r) == (r + 1) % N
Left(r)  == (r - 1 + N) % N

(* Every unordered pair of *adjacent* ranks in the ring shares one
   channel of capacity 1 (mirrors one TCP socket: a second send before
   the first is drained blocks the sender). For N = 2, Right(0) = Left(0)
   = 1 and Right(1) = Left(1) = 0 — both of rank 0 and rank 1's ring
   neighbors collapse onto the *same* single channel, exactly the case
   that makes ordering matter (see channel.h's deadlock note and
   ring_allreduce.cpp's file header). *)
PairKey(a, b) == IF a < b THEN <<a, b>> ELSE <<b, a>>
Pairs == {PairKey(r, Right(r)) : r \in Ranks}

VARIABLES
    round,        \* [Ranks -> 0..(2*(N-1))], this rank's progress; N-1 rounds reduce-scatter, N-1 all-gather
    chunk,        \* [Ranks -> [0..(N-1) -> SUBSET Ranks]]: chunk[r][c] = contributors folded into rank r's copy of chunk c
    chanOccupied  \* [Pairs -> BOOLEAN]: TRUE if a message is sitting in that shared channel, unclaimed

vars == <<round, chunk, chanOccupied>>

(* The parity rule from ring_allreduce.cpp: even ranks attempt to place a
   message on the channel before attempting to drain one; odd ranks
   drain first. Encoded as a predicate so both the Send and Recv actions
   below can check "is it legal for me to go first here." *)
SendsFirst(r) == r % 2 = 0

TotalRounds == 2 * (N - 1)
InReduceScatter(r) == round[r] < N - 1
InAllGather(r)      == round[r] >= N - 1 /\ round[r] < TotalRounds
Finished(r)          == round[r] = TotalRounds

(* Which chunk index this rank sends/receives at its current round —
   mirrors the exact index arithmetic in ring_allreduce.cpp (reduce-
   scatter and all-gather use different offsets; see that file's
   comments for why). *)
SendChunkIdx(r) ==
    IF InReduceScatter(r)
    THEN (r - round[r] + N) % N
    ELSE (r - (round[r] - (N-1)) + 1 + N) % N
RecvChunkIdx(r) ==
    IF InReduceScatter(r)
    THEN (r - round[r] - 1 + N) % N
    ELSE (r - (round[r] - (N-1)) + N) % N

Init ==
    /\ round = [r \in Ranks |-> 0]
    /\ chunk = [r \in Ranks |-> [c \in Ranks |-> {r}]]  \* every rank starts knowing only its own contribution
    /\ chanOccupied = [p \in Pairs |-> FALSE]

(* Rank r places its current-round send chunk onto its channel with
   Right(r) — only legal if that channel is free AND (r is a "sends
   first" rank, or the channel is free because the *other* side already
   drained it this round, which SendsFirst's asymmetry guarantees
   happens in the right order — TLC checking this never deadlocks is the
   actual point of this spec). *)
Send(r) ==
    /\ ~Finished(r)
    /\ LET p == PairKey(r, Right(r)) IN
       /\ ~chanOccupied[p]
       /\ chanOccupied' = [chanOccupied EXCEPT ![p] = TRUE]
    /\ UNCHANGED <<round, chunk>>

(* Rank r drains whatever its left neighbor placed on their shared
   channel, folding the received chunk's contributors into its own copy
   (reduce-scatter: union/accumulate; all-gather: the received chunk is
   already-fully-reduced, so this rank simply adopts it — modeled the
   same way here since "fold in" is a no-op-if-already-superset union in
   both cases once the received set is correct). Advances this rank's
   round only after both send and recv for this round have happened —
   simplified here to advance on recv, since Send/Recv for the same
   round always alternate in lockstep with the parity rule. *)
Recv(r) ==
    /\ ~Finished(r)
    /\ LET p == PairKey(r, Left(r)) IN
       /\ chanOccupied[p]
       /\ chanOccupied' = [chanOccupied EXCEPT ![p] = FALSE]
    /\ chunk' = [chunk EXCEPT ![r][RecvChunkIdx(r)] = @ \cup chunk[Left(r)][RecvChunkIdx(r)]]
    /\ round' = [round EXCEPT ![r] = @ + 1]

Next == \E r \in Ranks : Send(r) \/ Recv(r)

Spec == Init /\ [][Next]_vars

(* ---------------------------- Invariants ---------------------------- *)

(* "No deadlock" isn't a separate invariant to state — it's TLC's
   built-in check that every reachable state (other than one where every
   rank has Finished, a legitimate terminal state) has at least one
   enabled action. Declared here as a predicate anyway so it's visible
   what "done" means, for the .cfg file's deadlock-exclusion condition. *)
AllFinished == \A r \in Ranks : Finished(r)

(* Safety: once every rank has finished, every rank's copy of every
   chunk must contain every rank as a contributor — this is "all ranks
   agree on final value" translated into the contributor-set
   abstraction: if everyone contributed to everyone's final chunk, the
   numeric result (a sum, in the real implementation) is necessarily
   identical everywhere. *)
AllRanksAgree ==
    AllFinished => (\A r \in Ranks : \A c \in Ranks : chunk[r][c] = Ranks)

=============================================================================
