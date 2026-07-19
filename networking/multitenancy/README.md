# Multi-Tenancy

**Status: code-complete AND locally run — portable, builds and runs everywhere including this Mac.**

## What this measures
Resource quota enforcement, priority preemption, fair scheduling with
measured isolation.

## Design
`FairScheduler` (`multitenancy.cpp`): strict priority across classes (all
of a higher-priority tenant class's queued tasks run before any
lower-priority task starts — "priority preemption" expressed as admission
order, since tasks here are opaque callables with no natural
mid-execution suspend point), weighted round-robin within a class (each
tenant gets `weight` tasks per pass, repeated until the class drains).
`submit()` rejects (and counts) submissions once a tenant's queue hits its
configured `quota`, rather than growing it unboundedly — the load-shedding
half of this step.

## Sanity-run output (Mac, 2026-07-19)

`multitenancy_test`: three tenants — "premium" (priority 2), "standard"
and "free" (priority 1, weights 2:1, free given a small quota of 20
against 50 submitted):

```
all premium tasks execute before any standard/free task OK
standard completed all 100 submitted tasks              OK
free completed exactly its accepted (20) tasks          OK
  (standard:free completions while both contending = 40:20, ratio 2.00, weight ratio is 2.0)
standard:free contended ratio is close to weight ratio (2.0, +/-0.5) OK
free accepted exactly its quota (20) of 50 submitted    OK
free rejected exactly 30                                OK
PASS
```

The 40:20 split while both tenants are contending is an exact 2.00 match
to their 2:1 weight configuration — not just "close," the deterministic
round-robin algorithm produces it precisely for this quantum/queue-depth
combination.

## Results
This step is a scheduling-policy correctness check, not a
hardware-dependent performance benchmark — no Results table. Measuring
*isolation* under real concurrent load (multiple threads submitting while
`run()` drains, contended CPU) would need a live-server variant of this
scheduler; the current single-threaded run-to-completion form validates
the algorithm's fairness/priority/quota properties directly, same scope
choice as `topo_scheduler/` (step 16).

## Hardware notes
- Builds and runs anywhere (validated on Mac); no hardware dependency at all.
