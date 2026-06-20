#pragma once
#include <string>
// TODO: implement with OpenTelemetry C++ SDK

// Lightweight span wrapper for distributed tracing.
// Integrates with OTLP exporter → Jaeger or Tempo backend.
class Span {
public:
    Span(const std::string& name, const std::string& trace_id = "");
    ~Span();  // automatically ends span on destruction

    void set_attribute(const std::string& key, const std::string& value);
    void set_attribute(const std::string& key, double value);
    void add_event(const std::string& name);
    void set_error(const std::string& message);

    const std::string& trace_id() const;
    const std::string& span_id() const;
};

// Initialize OTel with OTLP HTTP exporter
void init_tracing(const std::string& service_name,
                  const std::string& otlp_endpoint = "http://localhost:4318");
