---------------------------- MODULE RaftMC ----------------------------
(* Model-checking wrapper around networking/tla_raft/Raft.tla, the
   standard TLA+ pattern for keeping a spec's own module free of
   model-specific bounding (a separate "MC" module extends the real spec
   and adds only what TLC needs to terminate) rather than editing
   Raft.tla itself to add a constraint the C++ implementation has no
   analog of.

   Raft.tla's Timeout action has no upper bound on currentTerm, and
   ClientRequest has no upper bound on log length — both correct and
   necessary for the real spec (a real cluster can run forever), but
   without SOME bound TLC's state space is infinite. StateConstraint
   below is that bound: small enough to terminate in reasonable time on
   a 3-server/2-value model, large enough to still explore multiple
   election terms and multi-entry logs (bounding too tightly would make
   e.g. OneLeaderPerTerm's cross-term case unreachable, silently proving
   nothing about it). *)

EXTENDS Raft

StateConstraint ==
    /\ \A s \in Server : currentTerm[s] <= 3
    /\ \A s \in Server : Len(log[s]) <= 3

=============================================================================
