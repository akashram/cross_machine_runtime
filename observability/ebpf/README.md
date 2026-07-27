# ebpf

**Status: code-complete, hardware/toolchain-gated — real BCC program,
unrun. No Linux kernel, no BCC, on this Mac; also needs `sudo`/
`CAP_SYS_ADMIN` at runtime.**

## What this measures

PLAN.md Phase 10 step 1: kernel scheduler (`sched_switch`,
`sched_wakeup`), memory subsystem (page faults), network stack
(`sock_sendmsg`, `tcp_retransmit`), visualized as timeline traces.

## Design

- `sched_probe.py`: four real BCC tracepoint/kprobe attachments, not
  commented-out pseudocode:
  - `sched_switch`: per-CPU context-switch timeline — scheduler jitter on
    a pinned hot-path thread's CPU shows up directly here, the
    observability counterpart to `cpu_engine`'s affinity work.
  - `sched_wakeup` joined with `sched_switch` on pid (via a `BPF_HASH`
    scratch table) to compute wakeup-to-running latency — neither
    tracepoint alone gives this; it's the gap between them.
  - Page faults via a kprobe/kretprobe pair on `handle_mm_fault`: the
    entry probe marks the pid, the return probe reads `VM_FAULT_MAJOR`
    off the return value to classify major vs. minor — that bit is only
    available at return, so a plain entry kprobe can't distinguish them.
  - `sock_sendmsg` / `tcp_retransmit_skb`: per-call send events and
    retransmits, the signal `networking/`'s collective/RDMA steps would
    want correlated against their own measured throughput dips.
- Output is JSON Lines, one event object per line — the same shape
  `observability/dashboard` (step 3) and `observability/opentelemetry`
  (step 2) are designed to ingest, so a real eBPF trace and a real OTel
  trace could be correlated on one timeline once both exist on real
  hardware.

## Results
TODO: run on Linux (kernel ≥ 5.8) with BCC installed
(`apt install bpfcc-tools`), `sudo python3 sched_probe.py --duration 10
--out trace.jsonl` against a real workload from this repo (e.g. one of
`networking/`'s multi-thread simulated-rank tests, or a real multi-node
run once hardware is provisioned).

| Metric | Value |
|--------|-------|
| Scheduler jitter (switch rate on a pinned CPU) | TODO |
| Wakeup-to-running p50/p99 latency | TODO |
| Major vs. minor fault rate | TODO |
| Retransmit rate during a collective op | TODO |

## Hardware notes
- Required: Linux kernel ≥ 5.8, BCC (`bpfcc-tools`), `sudo`/
  `CAP_SYS_ADMIN` (or `CAP_BPF` on newer kernels) at runtime.
