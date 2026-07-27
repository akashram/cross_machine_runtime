# opentelemetry

**Status: code-complete AND locally run — no OTel C++ SDK dependency, no
GPU/Linux dependency.**

## What this measures

PLAN.md Phase 10 step 2: distributed tracing across all nodes, spans for
every major operation, trace correlation with eBPF events (step 1).

## Design

- **Deliberate deviation from the original stub's plan** ("implement with
  OpenTelemetry C++ SDK"): hand-rolled instead, same rationale as this
  repo's other hand-rolled-instead-of-dependency choices
  (`foundation/proptest` instead of a property-testing library,
  `distributed_training::Matrix` instead of Eigen). The concepts this
  step needs — span lifecycle, 128-bit trace id / 64-bit span id,
  parent-child nesting, attributes, events, OTLP's JSON wire shape — are
  simple enough to implement directly and verify locally, which is what
  makes this step **code-complete AND actually run**, rather than needing
  a new SDK install just to sit unrun (the JAX/Phase-8 outcome).
- `Span`: RAII — constructor records start time and pushes itself onto a
  thread-local span stack (establishing `parent_span_id` implicitly from
  whatever span is currently on top, matching how OTel's own SDKs handle
  context propagation within a thread); destructor records end time, pops
  the stack, and hands the completed record to the active exporter.
- `JsonFileExporter`: appends one OTLP-JSON-shaped object per line —
  real, parseable output (`tracer_test.cpp` parses it back and checks
  every field), not a mocked/fake sink.
- **The one genuinely untestable-without-infrastructure piece, left as a
  documented gap rather than shipped unverified**: real OTLP-over-HTTP
  POST to a live Jaeger/Tempo collector. There's no collector running
  anywhere this repo can check the exchange against, so `init_tracing()`
  takes a `SpanExporter` (an interface `JsonFileExporter` implements)
  instead of an OTLP endpoint URL — a real HTTP exporter would implement
  the same interface, and the span lifecycle logic (the actually
  interesting part of this step) doesn't change either way.

## Results (captured 2026-07-27, Apple clang 14 / `-std=c++2b`, this Mac)

```
PASS  two spans exported (inner, then outer -- LIFO destruction order)
PASS  trace id is a 32-hex-char (128-bit) value, matching OTel's format
PASS  span id is a 16-hex-char (64-bit) value, matching OTel's format
PASS  nested span shares its parent's trace id
PASS  nested span's parentSpanId equals the outer span's spanId
PASS  root span has an empty parentSpanId
PASS  span name recorded correctly
PASS  service name propagated from init_tracing
PASS  one span exported
PASS  string attribute recorded
PASS  numeric attribute recorded
PASS  event recorded
PASS  error status recorded
PASS  error message recorded
PASS
```

## Hardware notes
None for the span lifecycle/JSON export tested here. A real OTLP/HTTP
exporter to a live Jaeger/Tempo collector needs that collector running
somewhere (Linux server, container, or cloud service) — out of scope for
this Mac, and orthogonal to the span logic itself (see Design).
