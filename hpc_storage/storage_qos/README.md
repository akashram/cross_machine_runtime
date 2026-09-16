# storage_qos

**Status: code-complete AND locally run — pure CPU, no external dependency.**

## What this measures

PLAN.md Phase 21 step 8: storage bandwidth/IOPS QoS, a "direct extension
of `networking/multitenancy`'s existing fairness/quota logic, reframed
for storage bandwidth/IOPS contention between simultaneous training jobs
sharing one backend, rather than network bandwidth between peers — same
mechanism, different contended resource." `StorageQosScheduler` composes
the real, UNMODIFIED `networking::multitenancy::FairScheduler` for
priority/weighted-round-robin admission and adds a per-tenant
bytes/sec token-bucket bandwidth limiter on top — the one genuinely new
mechanism multitenancy.h has no equivalent of (its existing quota is a
task COUNT, not a byte rate).

## Results (captured 2026-09-16, Apple clang 14, `--preset release`, this Mac)

```
== Storage QoS: bandwidth token-bucket gate ==

  tenant a: 500000 bytes admitted, 0 bytes throttled
PASS  tenant a's requests, all within its 1MB/s budget, were all admitted
  tenant b: 100000 bytes admitted, 100000 bytes throttled (after 2 requests, 100KB budget)
PASS  tenant b's first 100KB request exactly fits its 100KB/s budget and is admitted
PASS  tenant b's second 100KB request exceeds its now-exhausted budget and is throttled (not admitted) -- the token-bucket gate, the new mechanism this step adds on top of multitenancy's unchanged quota/priority logic
PASS  after tick(1.0) refills tenant b's bucket, an identical 100KB request that was just throttled is now admitted -- throttling is a rate limit, not a ban
PASS  drain() runs the composed, unmodified FairScheduler::run() without error

PASS
```

## Findings

- **The bandwidth token-bucket gate behaves exactly as a rate limiter
  should**: a tenant within budget is admitted every time (tenant a, 5x
  100KB requests inside a 1MB/s budget, 0 bytes throttled); a tenant that
  exceeds its budget is throttled (tenant b's second request), and the
  SAME request that was throttled is admitted once `tick(1.0)` refills
  the bucket — confirming this is a genuine rate limit, not a permanent
  rejection, the real distinction that matters for a live storage system
  (a throttled tenant should eventually make progress, not starve).
- **The composition works cleanly**: `FairScheduler`'s existing
  priority/task-count-quota admission gate (`submit()` returning false on
  quota overflow) and the new bandwidth gate are INDEPENDENT checks — a
  request must pass both to be admitted, exactly the "same mechanism,
  different contended resource" composition PLAN.md step 8 asks for,
  with zero modification to `multitenancy.h`/`.cpp`.

## Hardware notes

None — pure CPU, no real storage I/O in this step (the bandwidth budget
is a pure accounting simulation, same scope discipline as
`networking/multitenancy`'s own "the scheduling POLICY in isolation, not
a live server" design). See `storage_qos.h`'s header comment and
`hpc_storage/DESIGN.md`.
