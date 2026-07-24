// axi_stream_passthrough.v — hand-written RTL model of axi_stream/axi_passthrough.cpp
// for cocotb simulation.
//
// axi_stream/axi_passthrough.cpp is HLS C++ — turning it into simulatable
// RTL needs Vitis HLS, which isn't available on this Mac (see that step's
// README). This is a small hand-written Verilog module with the same
// behavior (copy in_stream to out_stream unchanged, respecting AXI4-Stream
// backpressure) so cocotb + Icarus Verilog has something real to drive
// without a Xilinx toolchain.
//
// Standard single-register AXI4-Stream slice: s_axis_tready is
// combinational (empty-or-draining-this-cycle), so throughput stays II=1
// under sustained backpressure-free traffic, and the registered
// m_axis_t* outputs hold their value whenever m_axis_tready is low —
// exactly the two rules ila_debug/axi_trace_checker.py checks
// (VALID-hold, data-stability).
`timescale 1ns / 1ps

module axi_stream_passthrough #(
    parameter DATA_WIDTH = 32
) (
    input  wire                    aclk,
    input  wire                    aresetn,

    input  wire [DATA_WIDTH-1:0]   s_axis_tdata,
    input  wire                    s_axis_tlast,
    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,

    output reg  [DATA_WIDTH-1:0]   m_axis_tdata,
    output reg                     m_axis_tlast,
    output reg                     m_axis_tvalid,
    input  wire                    m_axis_tready
);

    // Ready to accept a new input word whenever the output register is
    // empty (!m_axis_tvalid) or being drained this same cycle
    // (m_axis_tready) — the standard "register slice" ready equation.
    assign s_axis_tready = !m_axis_tvalid || m_axis_tready;

    always @(posedge aclk) begin
        if (!aresetn) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tdata  <= {DATA_WIDTH{1'b0}};
            m_axis_tlast  <= 1'b0;
        end else if (s_axis_tready) begin
            // Either loading a fresh word (s_axis_tvalid=1) or the
            // output register is draining with nothing behind it
            // (s_axis_tvalid=0), in which case tvalid correctly clears.
            m_axis_tvalid <= s_axis_tvalid;
            m_axis_tdata  <= s_axis_tdata;
            m_axis_tlast  <= s_axis_tlast;
        end
        // else (s_axis_tready == 0): output register is full and
        // downstream isn't ready — hold tdata/tlast/tvalid unchanged.
    end

endmodule
