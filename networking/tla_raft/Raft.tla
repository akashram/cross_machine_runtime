(* Raft.tla — TLA+ specification of Raft consensus *)
(* TODO: write full spec based on Ongaro's thesis TLA+ appendix *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Servers, Values

(* TODO: define state variables *)
(* VARIABLES currentTerm, votedFor, log, commitIndex, lastApplied, ... *)

(* TODO: define Init, Next, Spec *)
(* TODO: define safety invariants: OneLeaderPerTerm, LogMatching *)
(* TODO: define liveness: EventuallyCommits *)
