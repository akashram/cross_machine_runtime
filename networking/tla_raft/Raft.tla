------------------------------ MODULE Raft ------------------------------
(* TLA+ specification of Raft consensus (Ongaro & Ousterhout), scoped to
   match what networking/raft/raft.cpp actually implements: leader
   election and log replication. Cluster membership changes and log
   compaction/snapshotting are out of scope here, same as the C++ —
   documented in raft/raft.h and raft/README.md.

   This is checked in but NOT run through TLC in this session (no local
   Java/TLA+ toolchain used here — see networking/tla_raft/README.md).
   Written to the structure of the canonical raft.tla spec (Ongaro's
   PhD thesis appendix / the widely-used community TLA+ model), adapted
   down to this project's scope. *)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Server,        \* the set of server ids, e.g. {s1, s2, s3}
    Value,         \* the set of possible command values a client can propose
    Nil            \* a value not in Server, used for "no leader" / "voted for nobody"

ServerState == {"Follower", "Candidate", "Leader"}

VARIABLES
    currentTerm,   \* [s \in Server -> Nat]
    state,         \* [s \in Server -> ServerState]
    votedFor,      \* [s \in Server -> Server \cup {Nil}]
    log,           \* [s \in Server -> Seq([term: Nat, value: Value])]
    commitIndex,   \* [s \in Server -> Nat]
    votesGranted,  \* [s \in Server -> SUBSET Server] — who has voted for s in its current term
    nextIndex,     \* [s \in Server -> [Server -> Nat]] — leader's view, meaningful only while s is Leader
    matchIndex,    \* [s \in Server -> [Server -> Nat]]
    messages       \* a set of in-flight RPC messages (request or response records)

raftVars == <<currentTerm, state, votedFor, log, commitIndex, votesGranted, nextIndex, matchIndex, messages>>

Quorum == {q \in SUBSET Server : 2 * Cardinality(q) > Cardinality(Server)}

LastTerm(s) == IF Len(log[s]) = 0 THEN 0 ELSE log[s][Len(log[s])].term

(* ---- Message shapes (records, distinguished by "mtype") ---- *)
RequestVoteRequest(s, t) == [mtype |-> "RequestVoteRequest", msource |-> s, mterm |-> t,
                             mlastLogTerm |-> LastTerm(s), mlastLogIndex |-> Len(log[s])]

RequestVoteResponse(s, dest, t, granted) ==
    [mtype |-> "RequestVoteResponse", msource |-> s, mdest |-> dest, mterm |-> t, mvoteGranted |-> granted]

AppendEntriesRequest(s, dest, t, prevIdx, prevTerm, entries, commit) ==
    [mtype |-> "AppendEntriesRequest", msource |-> s, mdest |-> dest, mterm |-> t,
     mprevLogIndex |-> prevIdx, mprevLogTerm |-> prevTerm, mentries |-> entries, mcommitIndex |-> commit]

AppendEntriesResponse(s, dest, t, success, matchIdx) ==
    [mtype |-> "AppendEntriesResponse", msource |-> s, mdest |-> dest, mterm |-> t,
     msuccess |-> success, mmatchIndex |-> matchIdx]

(* ---------------------------- Init ---------------------------- *)
Init ==
    /\ currentTerm = [s \in Server |-> 0]
    /\ state        = [s \in Server |-> "Follower"]
    /\ votedFor     = [s \in Server |-> Nil]
    /\ log          = [s \in Server |-> << >>]
    /\ commitIndex  = [s \in Server |-> 0]
    /\ votesGranted = [s \in Server |-> {}]
    /\ nextIndex    = [s \in Server |-> [t \in Server |-> 1]]
    /\ matchIndex   = [s \in Server |-> [t \in Server |-> 0]]
    /\ messages     = {}

(* -------------------- Election actions -------------------- *)

(* A follower or candidate times out and starts a new election — the
   nondeterministic-timeout modeling standard to Raft TLA+ specs; the
   real ticker/randomized-timeout logic in raft.cpp is what makes this
   *eventually* happen without livelocking forever, which is a liveness
   property (see EventuallyElectsLeader below), not a safety one. *)
Timeout(s) ==
    /\ state[s] \in {"Follower", "Candidate"}
    /\ state'        = [state EXCEPT ![s] = "Candidate"]
    /\ currentTerm'  = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ votedFor'     = [votedFor EXCEPT ![s] = s]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = {s}]
    /\ messages' = messages \cup {RequestVoteRequest(s, currentTerm[s] + 1)}
    /\ UNCHANGED <<log, commitIndex, nextIndex, matchIndex>>

HandleRequestVoteRequest(dest, m) ==
    LET logOk == \/ m.mlastLogTerm > LastTerm(dest)
                 \/ /\ m.mlastLogTerm = LastTerm(dest)
                    /\ m.mlastLogIndex >= Len(log[dest])
        grant  == /\ m.mterm >= currentTerm[dest]
                  /\ logOk
                  /\ votedFor[dest] \in {Nil, m.msource}
    IN
    /\ m.mterm >= currentTerm[dest] \* stale requests are simply not responded to (modeled by requiring this)
    /\ currentTerm' = [currentTerm EXCEPT ![dest] = m.mterm]
    /\ state'       = [state EXCEPT ![dest] = IF m.mterm > currentTerm[dest] THEN "Follower" ELSE state[dest]]
    /\ votedFor'    = [votedFor EXCEPT ![dest] = IF grant THEN m.msource ELSE votedFor[dest]]
    /\ messages' = (messages \ {m}) \cup {RequestVoteResponse(dest, m.msource, m.mterm, grant)}
    /\ UNCHANGED <<log, commitIndex, votesGranted, nextIndex, matchIndex>>

HandleRequestVoteResponse(dest, m) ==
    /\ m.mterm = currentTerm[dest]
    /\ state[dest] = "Candidate"
    /\ votesGranted' = [votesGranted EXCEPT ![dest] =
                            IF m.mvoteGranted THEN votesGranted[dest] \cup {m.msource} ELSE votesGranted[dest]]
    /\ messages' = messages \ {m}
    /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex, nextIndex, matchIndex>>

(* Election Safety falls out of Quorum's definition (any two quorums
   intersect) plus votedFor being set at most once per term — TLC checks
   this is actually preserved by every action, not just this one. *)
BecomeLeader(s) ==
    /\ state[s] = "Candidate"
    /\ votesGranted[s] \in Quorum
    /\ state' = [state EXCEPT ![s] = "Leader"]
    /\ nextIndex'  = [nextIndex EXCEPT ![s] = [t \in Server |-> Len(log[s]) + 1]]
    /\ matchIndex' = [matchIndex EXCEPT ![s] = [t \in Server |-> 0]]
    /\ UNCHANGED <<currentTerm, votedFor, log, commitIndex, votesGranted, messages>>

(* -------------------- Replication actions -------------------- *)

ClientRequest(s, v) ==
    /\ state[s] = "Leader"
    /\ log' = [log EXCEPT ![s] = Append(log[s], [term |-> currentTerm[s], value |-> v])]
    /\ UNCHANGED <<currentTerm, state, votedFor, commitIndex, votesGranted, nextIndex, matchIndex, messages>>

SendAppendEntries(s, dest) ==
    LET prevIdx  == nextIndex[s][dest] - 1
        prevTerm == IF prevIdx = 0 THEN 0 ELSE log[s][prevIdx].term
        entries  == SubSeq(log[s], nextIndex[s][dest], Len(log[s]))
    IN
    /\ state[s] = "Leader"
    /\ dest # s
    /\ messages' = messages \cup
        {AppendEntriesRequest(s, dest, currentTerm[s], prevIdx, prevTerm, entries, commitIndex[s])}
    /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex, votesGranted, nextIndex, matchIndex>>

HandleAppendEntriesRequest(dest, m) ==
    LET logOk == \/ m.mprevLogIndex = 0
                 \/ /\ m.mprevLogIndex <= Len(log[dest])
                    /\ m.mprevLogIndex > 0
                    /\ log[dest][m.mprevLogIndex].term = m.mprevLogTerm
    IN
    /\ m.mterm >= currentTerm[dest]
    /\ currentTerm' = [currentTerm EXCEPT ![dest] = m.mterm]
    /\ state'       = [state EXCEPT ![dest] = "Follower"] \* a valid leader's AppendEntries always demotes a Candidate
    /\ IF logOk
       THEN /\ log' = [log EXCEPT ![dest] =
                        SubSeq(log[dest], 1, m.mprevLogIndex) \o m.mentries]
            /\ commitIndex' = [commitIndex EXCEPT ![dest] =
                                IF m.mcommitIndex > commitIndex[dest]
                                THEN Min({m.mcommitIndex, m.mprevLogIndex + Len(m.mentries)})
                                ELSE commitIndex[dest]]
            /\ messages' = (messages \ {m}) \cup
                {AppendEntriesResponse(dest, m.msource, m.mterm, TRUE, m.mprevLogIndex + Len(m.mentries))}
       ELSE /\ UNCHANGED <<log, commitIndex>>
            /\ messages' = (messages \ {m}) \cup {AppendEntriesResponse(dest, m.msource, m.mterm, FALSE, 0)}
    /\ UNCHANGED <<votedFor, votesGranted, nextIndex, matchIndex>>

HandleAppendEntriesResponse(dest, m) ==
    /\ state[dest] = "Leader"
    /\ m.mterm = currentTerm[dest]
    /\ IF m.msuccess
       THEN /\ nextIndex'  = [nextIndex  EXCEPT ![dest][m.msource] = m.mmatchIndex + 1]
            /\ matchIndex' = [matchIndex EXCEPT ![dest][m.msource] = m.mmatchIndex]
       ELSE /\ nextIndex'  = [nextIndex EXCEPT ![dest][m.msource] = Max({nextIndex[dest][m.msource] - 1, 1})]
            /\ UNCHANGED matchIndex
    /\ messages' = messages \ {m}
    /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex, votesGranted>>

(* Only commits an entry from the leader's *current* term by counting
   replicas directly — Raft's §5.4.2 rule. An older-term entry only ever
   becomes committed as a side effect of a later same-term entry
   committing. This is exactly the invariant raft.cpp's advanceCommitIndex
   comment calls out as "the single most common Raft implementation bug." *)
AdvanceCommitIndex(s) ==
    /\ state[s] = "Leader"
    /\ \E idx \in (commitIndex[s]+1)..Len(log[s]) :
        /\ log[s][idx].term = currentTerm[s]
        /\ {t \in Server : matchIndex[s][t] >= idx} \cup {s} \in Quorum
        /\ commitIndex' = [commitIndex EXCEPT ![s] = idx]
    /\ UNCHANGED <<currentTerm, state, votedFor, log, votesGranted, nextIndex, matchIndex, messages>>

(* ---------------------------- Next / Spec ---------------------------- *)
Next ==
    \/ \E s \in Server : Timeout(s)
    \/ \E s \in Server : BecomeLeader(s)
    \/ \E s \in Server, v \in Value : ClientRequest(s, v)
    \/ \E s, dest \in Server : SendAppendEntries(s, dest)
    \/ \E s \in Server : AdvanceCommitIndex(s)
    \/ \E m \in messages :
        \/ (m.mtype = "RequestVoteRequest" /\ HandleRequestVoteRequest(CHOOSE dest \in Server : TRUE, m))
        \/ (m.mtype = "RequestVoteResponse" /\ HandleRequestVoteResponse(m.mdest, m))
        \/ (m.mtype = "AppendEntriesRequest" /\ HandleAppendEntriesRequest(m.mdest, m))
        \/ (m.mtype = "AppendEntriesResponse" /\ HandleAppendEntriesResponse(m.mdest, m))

Spec == Init /\ [][Next]_raftVars

(* ---------------------------- Invariants ---------------------------- *)

(* Safety: at most one leader can be elected per term. This is the
   property Election Safety in the Raft paper (Figure 3), and the one
   BecomeLeader's Quorum-intersection argument is supposed to guarantee. *)
OneLeaderPerTerm ==
    \A s1, s2 \in Server :
        (state[s1] = "Leader" /\ state[s2] = "Leader" /\ currentTerm[s1] = currentTerm[s2]) => s1 = s2

(* Safety: if two logs contain an entry with the same index and term, the
   logs are identical in all entries up through that index (Raft paper's
   Log Matching property). *)
LogMatching ==
    \A s1, s2 \in Server :
        \A i \in 1..Min({Len(log[s1]), Len(log[s2])}) :
            (log[s1][i].term = log[s2][i].term) =>
                (SubSeq(log[s1], 1, i) = SubSeq(log[s2], 1, i))

(* Safety: a committed entry is present in the log of every future
   leader (Leader Completeness) — approximated here as: any two servers'
   committed prefixes agree. Full Leader Completeness needs a
   history/refinement argument beyond a single-state invariant; this
   version is what TLC can check directly against the state graph. *)
CommittedEntriesAgree ==
    \A s1, s2 \in Server :
        \A i \in 1..Min({commitIndex[s1], commitIndex[s2]}) :
            log[s1][i] = log[s2][i]

(* Liveness (needs weak fairness on the RPC-handling actions to be
   meaningful — WF_raftVars(Next) — otherwise TLC can construct an
   infinite behavior that just never delivers any message). Included for
   completeness; checking this is more expensive than the safety
   invariants above and TODO once this actually runs through TLC. *)
EventuallyCommits ==
    \A v \in Value :
        (\E s \in Server : \E i \in DOMAIN log[s] : log[s][i].value = v) ~>
            (\E s \in Server : \E i \in 1..commitIndex[s] : log[s][i].value = v)

=============================================================================
