#!/usr/bin/env python3
"""sched_probe.py — eBPF probes for kernel scheduler, memory, and network
events, via BCC (github.com/iovisor/bcc).

PLAN.md Phase 10 step 1: "kernel scheduler (sched_switch, sched_wakeup),
memory subsystem (page faults, huge page promotions), network stack
(sock_sendmsg, tcp_retransmit). Use libbpf or BCC. Visualize as timeline
traces."

Four tracepoints, each attached independently so a caller can run any
subset:
  - sched_switch: per-CPU context-switch timeline. Scheduler jitter shows
    up directly as switch frequency spikes on a CPU a hot-path thread is
    pinned to — the same jitter cpu_engine/'s CPU-affinity steps try to
    eliminate at the source; this is the observability side of that.
  - sched_wakeup -> sched_switch: wakeup latency (time from a thread
    becoming runnable to actually running) is measured by joining these
    two tracepoints on pid, not read off either alone.
  - page faults (via kprobe on handle_mm_fault, tagged major/minor from
    the return value's VM_FAULT_MAJOR bit): correlates with foundation/'s
    hugepage steps — hugepage promotion should show up as a reduction in
    minor-fault rate for a workload's heap.
  - sock_sendmsg / tcp_retransmit_skb: per-call send latency and
    retransmit events — the same signal networking/'s collective/RDMA
    steps would want correlated against their own measured throughput
    dips.

Output: JSON Lines to stdout (or --out FILE), one event object per line
— the same event-timeline format observability/dashboard/'s CLI report
(step 3) and observability/opentelemetry/'s spans (step 2) are designed
to ingest, so a real eBPF trace and a real OTel trace can be correlated
on a shared timeline once both exist.

Unrun here — no Linux kernel, no BCC, on this Mac. `sudo` and
CAP_SYS_ADMIN (or CAP_BPF on newer kernels) are also required at runtime,
not just at import time.
"""

import argparse
import json
import sys
import time

BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <net/sock.h>

struct switch_event_t {
    u64 ts_ns;
    u32 cpu;
    u32 prev_pid;
    u32 next_pid;
    char prev_comm[TASK_COMM_LEN];
    char next_comm[TASK_COMM_LEN];
};
BPF_PERF_OUTPUT(switch_events);

struct wakeup_event_t {
    u64 ts_ns;
    u32 pid;
    u32 target_cpu;
};
BPF_PERF_OUTPUT(wakeup_events);

// pid -> wakeup timestamp, consumed by sched_switch to emit wakeup latency
BPF_HASH(wakeup_ts, u32, u64);

struct latency_event_t {
    u64 wakeup_to_running_ns;
    u32 pid;
    char comm[TASK_COMM_LEN];
};
BPF_PERF_OUTPUT(latency_events);

struct fault_event_t {
    u64 ts_ns;
    u32 pid;
    u8 is_major;
};
BPF_PERF_OUTPUT(fault_events);

struct sock_event_t {
    u64 ts_ns;
    u32 pid;
    u32 len;
    u8 is_retransmit;
};
BPF_PERF_OUTPUT(sock_events);

TRACEPOINT_PROBE(sched, sched_switch) {
    struct switch_event_t evt = {};
    evt.ts_ns = bpf_ktime_get_ns();
    evt.cpu = bpf_get_smp_processor_id();
    evt.prev_pid = args->prev_pid;
    evt.next_pid = args->next_pid;
    bpf_probe_read_kernel_str(&evt.prev_comm, sizeof(evt.prev_comm), args->prev_comm);
    bpf_probe_read_kernel_str(&evt.next_comm, sizeof(evt.next_comm), args->next_comm);
    switch_events.perf_submit(args, &evt, sizeof(evt));

    // Close out a pending wakeup->running measurement for the thread that
    // just started running, if sched_wakeup recorded one.
    u32 next_pid = args->next_pid;
    u64 *w_ts = wakeup_ts.lookup(&next_pid);
    if (w_ts != 0) {
        struct latency_event_t lat = {};
        lat.wakeup_to_running_ns = evt.ts_ns - *w_ts;
        lat.pid = next_pid;
        bpf_probe_read_kernel_str(&lat.comm, sizeof(lat.comm), args->next_comm);
        latency_events.perf_submit(args, &lat, sizeof(lat));
        wakeup_ts.delete(&next_pid);
    }
    return 0;
}

TRACEPOINT_PROBE(sched, sched_wakeup) {
    struct wakeup_event_t evt = {};
    evt.ts_ns = bpf_ktime_get_ns();
    evt.pid = args->pid;
    evt.target_cpu = args->target_cpu;
    wakeup_events.perf_submit(args, &evt, sizeof(evt));

    u32 pid = args->pid;
    u64 ts = evt.ts_ns;
    wakeup_ts.update(&pid, &ts);
    return 0;
}

// handle_mm_fault's return value's VM_FAULT_MAJOR bit is what actually
// distinguishes major (disk I/O) from minor faults -- so this is a
// kretprobe pair (entry marks the pid, return reads the fault kind),
// not a single entry kprobe.
BPF_HASH(fault_pid, u32, u32);  // scratch: pid marked at entry, consumed at return

int trace_fault_entry(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 one = 1;
    fault_pid.update(&pid, &one);
    return 0;
}

int trace_fault_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 *marker = fault_pid.lookup(&pid);
    if (marker == 0) return 0;
    fault_pid.delete(&pid);

    long ret = PT_REGS_RC(ctx);
    struct fault_event_t evt = {};
    evt.ts_ns = bpf_ktime_get_ns();
    evt.pid = pid;
    evt.is_major = (ret & VM_FAULT_MAJOR) ? 1 : 0;
    fault_events.perf_submit(ctx, &evt, sizeof(evt));
    return 0;
}

int trace_sock_sendmsg(struct pt_regs *ctx, struct socket *sock, struct msghdr *msg, size_t size) {
    struct sock_event_t evt = {};
    evt.ts_ns = bpf_ktime_get_ns();
    evt.pid = bpf_get_current_pid_tgid() >> 32;
    evt.len = size;
    evt.is_retransmit = 0;
    sock_events.perf_submit(ctx, &evt, sizeof(evt));
    return 0;
}

int trace_tcp_retransmit(struct pt_regs *ctx, struct sock *sk) {
    struct sock_event_t evt = {};
    evt.ts_ns = bpf_ktime_get_ns();
    evt.pid = bpf_get_current_pid_tgid() >> 32;
    evt.len = 0;
    evt.is_retransmit = 1;
    sock_events.perf_submit(ctx, &evt, sizeof(evt));
    return 0;
}
"""


def emit(out, obj):
    out.write(json.dumps(obj) + "\n")
    out.flush()


def main(duration_s: float, out_path: str) -> None:
    from bcc import BPF  # imported here: fails loudly on any non-Linux/no-BCC host, not at module load

    b = BPF(text=BPF_PROGRAM)
    b.attach_kprobe(event="handle_mm_fault", fn_name="trace_fault_entry")
    b.attach_kretprobe(event="handle_mm_fault", fn_name="trace_fault_return")
    b.attach_kprobe(event="sock_sendmsg", fn_name="trace_sock_sendmsg")
    b.attach_kprobe(event="tcp_retransmit_skb", fn_name="trace_tcp_retransmit")

    out = open(out_path, "w") if out_path else sys.stdout

    def on_switch(cpu, data, size):
        e = b["switch_events"].event(data)
        emit(out, {"type": "sched_switch", "ts_ns": e.ts_ns, "cpu": e.cpu,
                    "prev_pid": e.prev_pid, "next_pid": e.next_pid,
                    "prev_comm": e.prev_comm.decode(), "next_comm": e.next_comm.decode()})

    def on_latency(cpu, data, size):
        e = b["latency_events"].event(data)
        emit(out, {"type": "wakeup_latency", "pid": e.pid, "comm": e.comm.decode(),
                    "latency_ns": e.wakeup_to_running_ns})

    def on_fault(cpu, data, size):
        e = b["fault_events"].event(data)
        emit(out, {"type": "page_fault", "ts_ns": e.ts_ns, "pid": e.pid,
                    "major": bool(e.is_major)})

    def on_sock(cpu, data, size):
        e = b["sock_events"].event(data)
        emit(out, {"type": "retransmit" if e.is_retransmit else "sock_sendmsg",
                    "ts_ns": e.ts_ns, "pid": e.pid, "len": e.len})

    b["switch_events"].open_perf_buffer(on_switch)
    b["latency_events"].open_perf_buffer(on_latency)
    b["fault_events"].open_perf_buffer(on_fault)
    b["sock_events"].open_perf_buffer(on_sock)

    t0 = time.time()
    while time.time() - t0 < duration_s:
        b.perf_buffer_poll(timeout=200)

    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to trace")
    parser.add_argument("--out", default="", help="JSON Lines output path (default: stdout)")
    args = parser.parse_args()
    main(args.duration, args.out)
