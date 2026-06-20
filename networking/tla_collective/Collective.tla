(* Collective.tla — TLA+ spec for ring all-reduce correctness *)
(* TODO: formalize ring all-reduce protocol and verify with TLC *)

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS N  \* number of ranks

(* TODO: define ring send/recv transitions, consistency invariant *)
