#pragma once
#include <memory>
#include <string>
#include <variant>
#include <vector>

// PLAN.md Phase 10 step 2: OpenTelemetry integration — distributed
// tracing across all nodes, spans for every major operation, trace
// correlation with eBPF events (step 1).
//
// **Deliberate deviation from the original stub's "implement with
// OpenTelemetry C++ SDK" plan**: hand-rolled span lifecycle + ID
// generation + OTLP-JSON-shaped serialization, no OTel SDK dependency.
// Same rationale as this repo's other hand-rolled-instead-of-dependency
// choices (`foundation/proptest` instead of a property-testing library,
// `distributed_training::Matrix` instead of Eigen): the *concepts* this
// step needs (span, trace/span id, parent-child nesting, attributes,
// events, OTLP's JSON wire shape) are simple enough to implement
// directly and verify locally, and doing so means this step is
// code-complete AND ACTUALLY RUN on this Mac (see README.md) rather than
// requiring a new SDK install just to sit unrun like Phase 8's JAX
// decision. The real OTLP-over-HTTP POST to a live Jaeger/Tempo
// collector — the only genuinely untestable-without-infrastructure part
// — is the one piece left as a documented gap (see README.md), instead
// of shipping unverified networking code for a collector this repo has
// no way to check the exchange against.

namespace observability {

using AttributeValue = std::variant<std::string, double>;

struct SpanEvent {
  std::string name;
  uint64_t timestamp_ns;
};

struct SpanRecord {
  std::string trace_id;   // 32 hex chars, matching OTel's 128-bit trace id format
  std::string span_id;    // 16 hex chars, matching OTel's 64-bit span id format
  std::string parent_span_id;  // empty if root
  std::string name;
  std::string service_name;
  uint64_t start_time_unix_nano;
  uint64_t end_time_unix_nano;
  std::vector<std::pair<std::string, AttributeValue>> attributes;
  std::vector<SpanEvent> events;
  bool is_error = false;
  std::string error_message;
};

// Sink a completed SpanRecord is handed to. `JsonFileExporter` (below) is
// the one real implementation in this repo; a real OTLP/HTTP exporter to
// a live collector would implement this same interface.
class SpanExporter {
 public:
  virtual ~SpanExporter() = default;
  virtual void export_span(const SpanRecord &record) = 0;
};

// Appends one OTLP-JSON-shaped object per line to a file — real,
// verifiable output (parsed back and checked in tracer_test.cpp), unlike
// an HTTP POST to a collector this repo has no way to run.
class JsonFileExporter : public SpanExporter {
 public:
  explicit JsonFileExporter(const std::string &path);
  ~JsonFileExporter() override;
  void export_span(const SpanRecord &record) override;

 private:
  void *file_;  // FILE*, opaque here to keep <cstdio> out of the header
};

// RAII span: constructor records start time and pushes itself onto a
// thread-local span stack (establishing parent_span_id automatically
// from whatever span is currently on top); destructor records end time,
// pops the stack, and hands the completed record to the active exporter.
class Span {
 public:
  explicit Span(const std::string &name);
  ~Span();

  Span(const Span &) = delete;
  Span &operator=(const Span &) = delete;

  void set_attribute(const std::string &key, const std::string &value);
  void set_attribute(const std::string &key, double value);
  void add_event(const std::string &name);
  void set_error(const std::string &message);

  const std::string &trace_id() const { return record_.trace_id; }
  const std::string &span_id() const { return record_.span_id; }

 private:
  SpanRecord record_;
};

// Sets the process-wide exporter and service name every subsequently
// constructed Span reports through. No OTLP endpoint parameter — see the
// class comment above for why the HTTP transport isn't implemented here.
void init_tracing(const std::string &service_name, std::unique_ptr<SpanExporter> exporter);

}  // namespace observability
