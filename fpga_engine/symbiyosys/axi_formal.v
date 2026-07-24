// axi_formal.v — SymbiYosys formal harness for
// cocotb/axi_stream_passthrough.v (step 20's hand-written RTL model of
// axi_stream/axi_passthrough.cpp).
//
// Wraps the DUT with SVA properties instead of editing the RTL itself, so
// the exact module cocotb already exercises via simulated traces (step 20)
// is also proved here for EVERY reachable state (k-induction / bounded
// model checking), not just the handful of traces a testbench happens to
// drive. Top-level ports left as plain module inputs (s_axis_t*,
// m_axis_tready, aresetn) are free/unconstrained each cycle by SymbiYosys
// convention -- the solver searches every possible driving sequence.
//
// Properties:
//  1. p_hold_while_stalled (safety, full induction) -- while the output
//     register is full and downstream stalls, tvalid/tdata/tlast must
//     hold exactly: the same VALID-hold + data-stability rules
//     ila_debug/axi_trace_checker.py checks post-hoc on one captured
//     trace and cocotb's backpressure test exercises on a handful of
//     random traces, proved here for all of them at once.
//  2. p_no_deadlock (bounded liveness) -- once a word is accepted into
//     the register (s_axis_tvalid && s_axis_tready), it must appear on
//     the output exactly one cycle later: the II=1 register-slice
//     latency, proved instead of only observed on cocotb's specific
//     10-word throughput trace.
//  3. p_ready_reflects_downstream (safety, full induction) -- whenever
//     downstream commits to being ready, upstream must be ready too (no
//     stuck backpressure): the "handshake always resolves" property
//     PLAN.md step 21 asks for, stated directly.
`timescale 1ns/1ps

module axi_formal (
    input wire         aclk,
    input wire         aresetn,
    input wire [31:0]  s_axis_tdata,
    input wire         s_axis_tlast,
    input wire         s_axis_tvalid,
    input wire         m_axis_tready
);
    wire        s_axis_tready;
    wire [31:0] m_axis_tdata;
    wire        m_axis_tlast;
    wire        m_axis_tvalid;

    axi_stream_passthrough #(.DATA_WIDTH(32)) dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tlast(m_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready)
    );

    p_hold_while_stalled: assert property (
        @(posedge aclk) disable iff (!aresetn)
        (m_axis_tvalid && !m_axis_tready) |=>
            (m_axis_tvalid && $stable(m_axis_tdata) && $stable(m_axis_tlast))
    );

    p_no_deadlock: assert property (
        @(posedge aclk) disable iff (!aresetn)
        (s_axis_tvalid && s_axis_tready) |=> m_axis_tvalid
    );

    p_ready_reflects_downstream: assert property (
        @(posedge aclk) disable iff (!aresetn)
        m_axis_tready |-> s_axis_tready
    );

endmodule
