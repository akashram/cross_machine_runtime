// tracer_test.cpp — verifies the real, exported JSON output: correct
// trace/span id formats, parent-child linkage across nested spans,
// attribute/event/error recording. Parses the JsonFileExporter's actual
// file output back (simple field-extraction, not a full JSON parser —
// sufficient since this file also controls the exact format emitted).
#include "tracer.h"

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

using namespace observability;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

std::vector<std::string> read_lines(const std::string &path) {
  std::ifstream f(path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(f, line))
    if (!line.empty()) lines.push_back(line);
  return lines;
}

// Extracts a quoted-string field's value: "key":"value" -> value.
std::string extract_string_field(const std::string &json, const std::string &key) {
  std::string needle = "\"" + key + "\":\"";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return "";
  pos += needle.size();
  auto end = json.find('"', pos);
  return json.substr(pos, end - pos);
}

bool is_hex(const std::string &s, std::size_t expected_len) {
  if (s.size() != expected_len) return false;
  for (char c : s)
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

void test_root_and_nested_span_linkage(const std::string &trace_path) {
  init_tracing("test_service", std::make_unique<JsonFileExporter>(trace_path));

  {
    Span outer("outer_op");
    {
      Span inner("inner_op");
      (void)inner;
    }  // inner destructs here, exported first
  }  // outer destructs here, exported second

  auto lines = read_lines(trace_path);
  require(lines.size() == 2, "two spans exported (inner, then outer -- LIFO destruction order)");
  if (lines.size() != 2) return;

  std::string inner_trace = extract_string_field(lines[0], "traceId");
  std::string inner_span = extract_string_field(lines[0], "spanId");
  std::string inner_parent = extract_string_field(lines[0], "parentSpanId");
  std::string outer_trace = extract_string_field(lines[1], "traceId");
  std::string outer_span = extract_string_field(lines[1], "spanId");
  std::string outer_parent = extract_string_field(lines[1], "parentSpanId");

  require(is_hex(inner_trace, 32), "trace id is a 32-hex-char (128-bit) value, matching OTel's format");
  require(is_hex(inner_span, 16), "span id is a 16-hex-char (64-bit) value, matching OTel's format");
  require(inner_trace == outer_trace, "nested span shares its parent's trace id");
  require(inner_parent == outer_span, "nested span's parentSpanId equals the outer span's spanId");
  require(outer_parent.empty(), "root span has an empty parentSpanId");
  require(extract_string_field(lines[0], "name") == "inner_op", "span name recorded correctly");
  require(extract_string_field(lines[1], "serviceName") == "test_service", "service name propagated from init_tracing");
}

void test_attributes_events_error(const std::string &trace_path) {
  init_tracing("attr_test_service", std::make_unique<JsonFileExporter>(trace_path));
  {
    Span s("attributed_op");
    s.set_attribute("backend", "cpu");
    s.set_attribute("latency_ms", 4.5);
    s.add_event("cache_miss");
    s.set_error("simulated failure");
  }

  auto lines = read_lines(trace_path);
  require(lines.size() == 1, "one span exported");
  if (lines.empty()) return;
  const std::string &json = lines[0];

  require(json.find("\"key\":\"backend\",\"value\":\"cpu\"") != std::string::npos, "string attribute recorded");
  require(json.find("\"key\":\"latency_ms\",\"value\":4.5") != std::string::npos, "numeric attribute recorded");
  require(json.find("\"name\":\"cache_miss\"") != std::string::npos, "event recorded");
  require(json.find("\"error\":true") != std::string::npos, "error status recorded");
  require(json.find("simulated failure") != std::string::npos, "error message recorded");
}

}  // namespace

int main() {
  std::string dir = fs::temp_directory_path().string() + "/otel_tracer_test_" + std::to_string(::getpid());
  fs::create_directories(dir);
  test_root_and_nested_span_linkage(dir + "/linkage.jsonl");
  test_attributes_events_error(dir + "/attrs.jsonl");
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
