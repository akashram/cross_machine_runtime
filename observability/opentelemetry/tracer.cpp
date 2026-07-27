#include "tracer.h"

#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <sstream>
#include <vector>

namespace observability {

namespace {

uint64_t now_unix_nanos() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
}

// Real random 128-bit trace id / 64-bit span id, hex-encoded to the
// exact width OTel's wire format uses (32 / 16 hex chars) -- not a
// counter or a hash of something predictable, since OTel ids are meant
// to be collision-resistant across an entire distributed trace.
std::mt19937_64 &rng() {
  static thread_local std::mt19937_64 engine(std::random_device{}());
  return engine;
}

std::string random_hex(int num_bytes) {
  std::uniform_int_distribution<int> byte_dist(0, 255);
  static const char *hex_digits = "0123456789abcdef";
  std::string out;
  out.reserve(static_cast<std::size_t>(num_bytes) * 2);
  for (int i = 0; i < num_bytes; ++i) {
    int b = byte_dist(rng());
    out += hex_digits[(b >> 4) & 0xF];
    out += hex_digits[b & 0xF];
  }
  return out;
}

std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

struct GlobalState {
  std::mutex mu;
  std::string service_name = "unknown_service";
  std::unique_ptr<SpanExporter> exporter;
};

GlobalState &global_state() {
  static GlobalState state;
  return state;
}

// Per-thread span stack: the top establishes the parent for a newly
// constructed Span, giving correct nesting without any explicit
// parent-passing at call sites (matches how OTel's C++/other SDKs handle
// implicit context propagation within a thread).
std::vector<Span *> &span_stack() {
  static thread_local std::vector<Span *> stack;
  return stack;
}

}  // namespace

void init_tracing(const std::string &service_name, std::unique_ptr<SpanExporter> exporter) {
  GlobalState &g = global_state();
  std::lock_guard<std::mutex> lock(g.mu);
  g.service_name = service_name;
  g.exporter = std::move(exporter);
}

Span::Span(const std::string &name) {
  record_.name = name;
  record_.start_time_unix_nano = now_unix_nanos();

  auto &stack = span_stack();
  if (!stack.empty()) {
    record_.trace_id = stack.back()->record_.trace_id;
    record_.parent_span_id = stack.back()->record_.span_id;
  } else {
    record_.trace_id = random_hex(16);  // 128 bits
  }
  record_.span_id = random_hex(8);  // 64 bits

  {
    GlobalState &g = global_state();
    std::lock_guard<std::mutex> lock(g.mu);
    record_.service_name = g.service_name;
  }

  stack.push_back(this);
}

Span::~Span() {
  record_.end_time_unix_nano = now_unix_nanos();

  auto &stack = span_stack();
  if (!stack.empty() && stack.back() == this) stack.pop_back();

  GlobalState &g = global_state();
  std::lock_guard<std::mutex> lock(g.mu);
  if (g.exporter) g.exporter->export_span(record_);
}

void Span::set_attribute(const std::string &key, const std::string &value) {
  record_.attributes.emplace_back(key, AttributeValue{value});
}

void Span::set_attribute(const std::string &key, double value) {
  record_.attributes.emplace_back(key, AttributeValue{value});
}

void Span::add_event(const std::string &name) { record_.events.push_back({name, now_unix_nanos()}); }

void Span::set_error(const std::string &message) {
  record_.is_error = true;
  record_.error_message = message;
}

JsonFileExporter::JsonFileExporter(const std::string &path) { file_ = std::fopen(path.c_str(), "w"); }

JsonFileExporter::~JsonFileExporter() {
  if (file_) std::fclose(static_cast<std::FILE *>(file_));
}

void JsonFileExporter::export_span(const SpanRecord &r) {
  if (!file_) return;
  auto *f = static_cast<std::FILE *>(file_);

  std::ostringstream oss;
  oss << "{";
  oss << "\"traceId\":\"" << r.trace_id << "\",";
  oss << "\"spanId\":\"" << r.span_id << "\",";
  oss << "\"parentSpanId\":\"" << r.parent_span_id << "\",";
  oss << "\"name\":\"" << json_escape(r.name) << "\",";
  oss << "\"serviceName\":\"" << json_escape(r.service_name) << "\",";
  oss << "\"startTimeUnixNano\":" << r.start_time_unix_nano << ",";
  oss << "\"endTimeUnixNano\":" << r.end_time_unix_nano << ",";

  oss << "\"attributes\":[";
  for (std::size_t i = 0; i < r.attributes.size(); ++i) {
    if (i) oss << ",";
    const auto &[key, value] = r.attributes[i];
    oss << "{\"key\":\"" << json_escape(key) << "\",\"value\":";
    if (std::holds_alternative<std::string>(value))
      oss << "\"" << json_escape(std::get<std::string>(value)) << "\"";
    else
      oss << std::get<double>(value);
    oss << "}";
  }
  oss << "],";

  oss << "\"events\":[";
  for (std::size_t i = 0; i < r.events.size(); ++i) {
    if (i) oss << ",";
    oss << "{\"name\":\"" << json_escape(r.events[i].name) << "\",\"timeUnixNano\":" << r.events[i].timestamp_ns
        << "}";
  }
  oss << "],";

  oss << "\"status\":{\"error\":" << (r.is_error ? "true" : "false") << ",\"message\":\""
      << json_escape(r.error_message) << "\"}";
  oss << "}\n";

  std::fputs(oss.str().c_str(), f);
  std::fflush(f);
}

}  // namespace observability
