# Phase 10: Observability, Testing, AI Integration

**Status: CODE COMPLETE (9/9 steps, 2026-07-27). Like Phase 9, most steps
turned out to have a genuinely local-runnable half — either the whole
step (dashboard, chaos's Raft scenario) or a portable core (opentelemetry,
the three AI-agent steps' heuristic/parsing halves) — and those are
**actually run**, real captured output in each step's own README, not
just written-but-unrun. `symbiyosys_ci` is a special case: it wraps
already-installed local tools (yosys/sby from Phase 7) and is fully run.
Genuinely gated: `ebpf` (Linux kernel), `tlc` (Java, declined for now —
see project memory), and the three AI agents' actual Claude API calls
(written as real client code, deliberately never invoked this session —
see project memory).**

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| 1 | ebpf | Kernel scheduler + memory + network eBPF probes | code-complete, hardware-gated (real BCC program, unrun) |
| 2 | opentelemetry | Distributed tracing across all nodes | **code-complete, run locally** (hand-rolled, no OTel SDK — see Design) |
| 3 | dashboard | CLI report: latency/GPU/FPGA/memory per rank | **code-complete, run locally** (synthetic-but-representative input) |
| 4 | tlc | TLC model checker for all TLA+ specs | code-complete, toolchain-gated (real MC configs for Raft + Collective, no Java — declined, see memory) |
| 5 | symbiyosys_ci | SymbiYosys on every FPGA RTL change | **code-complete, run locally** (yosys/sby already installed from Phase 7) |
| 6 | chaos | Automated fault injection + recovery suite | **2/3 scenarios run locally** (Raft leader-kill, FPGA thermal); GPU-node-kill stays a documented gap |
| 7 | nsight_agent | Nsight profile analyzer agent (Claude API) | parser **run locally**; real Claude API client code, never invoked |
| 8 | kernel_variant_agent | Kernel variant generator agent (Claude API) | real Claude API + nvcc/CUDA-event client code, never invoked (hardware+API-gated) |
| 9 | llm_autotune | LLM autotuning agent for placement decisions | heuristic core **run locally**; real Claude API client code, never invoked |

## Hardware notes
- eBPF: Linux with kernel ≥ 5.8 + BCC (`bpfcc-tools`), plus `sudo`/`CAP_SYS_ADMIN` at runtime.
- TLC: any OS, needs a Java runtime (not installed here — deferred, see project memory alongside Phase 8's JAX decision).
- symbiyosys_ci: none — `yosys`/`sby`/`z3` already installed locally (`~/oss-cad-suite`).
- AI agents (steps 7-9): `ANTHROPIC_API_KEY` (or `ant auth login`) for the Claude API calls, which this repo deliberately never makes; step 8's compile/benchmark half additionally needs `nvcc` + a CUDA GPU.

## Next
Hardware/toolchain validation (ebpf, tlc, the AI agents' actual API
calls, step 8's GPU half, chaos's remaining GPU-node-kill scenario) is
deferred along with every earlier phase — see the root `CLAUDE.md`'s
execution strategy. Next local-implementation phase: Phase 12 (Machine
Learning Library), currently stubbed — the last phase in the local
implementation order.
