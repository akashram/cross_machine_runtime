// dma_formal.v — SymbiYosys formal harness for
// cocotb/dma_controller.v (step 20's hand-written single-outstanding DMA
// engine, fixed there for a real one-cycle-early mem_rdata sampling bug --
// see that file's header).
//
// Proves the property test_dma_no_overlapping_transactions
// (cocotb/test_dma.py) only spot-checks at runtime for the specific
// traces it happens to drive: mem_rden and mem_wren are never asserted in
// the same cycle, for every reachable FSM state, not just the ones a
// testbench exercises. This holds by construction -- S_READ/S_READ_WAIT1/
// S_READ_WAIT2 only ever drive mem_rden, S_WRITE only ever drives
// mem_wren, and the FSM has no path that lands in both at once -- but
// "holds by construction" is exactly the kind of claim formal verification
// exists to machine-check instead of trusting by inspection.
//
// Control inputs (start, src_addr, dst_addr, length) are left free
// (SymbiYosys convention for unconstrained top-level ports), so the
// solver can start a new transfer at any point, with any address/length,
// including while a previous transfer is still in flight -- every
// scenario a cocotb trace would have to be deliberately written to hit.
`timescale 1ns/1ps

module dma_formal (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] src_addr,
    input  wire [15:0] dst_addr,
    input  wire [15:0] length,
    input  wire [31:0] mem_rdata
);
    wire        busy;
    wire        done;
    wire [15:0] mem_addr;
    wire [31:0] mem_wdata;
    wire        mem_rden;
    wire        mem_wren;

    dma_controller #(.ADDR_WIDTH(16), .DATA_WIDTH(32), .LEN_WIDTH(16)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .src_addr(src_addr), .dst_addr(dst_addr), .length(length),
        .busy(busy), .done(done),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata),
        .mem_rden(mem_rden), .mem_wren(mem_wren), .mem_rdata(mem_rdata)
    );

    p_no_overlapping_transactions: assert property (
        @(posedge clk) disable iff (!rst_n)
        !(mem_rden && mem_wren)
    );

endmodule
