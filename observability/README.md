# Phase 10: Observability, Testing, AI Integration

**Status: STUB — eBPF requires Linux; AI agent steps require Anthropic API key.**

## Steps

| # | Directory | What |
|---|-----------|------|
| 1 | ebpf | Kernel scheduler + memory + network eBPF probes |
| 2 | opentelemetry | Distributed tracing across all nodes |
| 3 | dashboard | CLI report: latency/GPU/FPGA/memory per rank |
| 4 | tlc | TLC model checker for all TLA+ specs |
| 5 | symbiyosys_ci | SymbiYosys on every FPGA RTL change |
| 6 | chaos | Automated fault injection + recovery suite |
| 7 | nsight_agent | Nsight profile analyzer agent (Claude API) |
| 8 | kernel_variant_agent | Kernel variant generator agent (Claude API) |
| 9 | llm_autotune | LLM autotuning agent for placement decisions |

## Hardware notes
- eBPF: Linux with kernel ≥ 5.8 + libbpf or BCC
- TLC: Java, any OS
- AI agents: Anthropic API key + GPU instance for kernel benchmarking
