// dma_controller.v — hand-written single-outstanding DMA engine for cocotb
// simulation (and step 21's SymbiYosys "never issues overlapping
// transactions" proof, which targets this exact module).
//
// dma/dma_orchestration.cpp (step 11) is host-side XRT code that talks to
// an already-synthesized kernel; this is the RTL-level DMA engine itself
// — read `length` words starting at `src_addr`, write each to
// `dst_addr` + offset, one word fully in flight at a time (read
// completes and is buffered before the corresponding write is issued),
// against a simple synchronous single-port memory interface. "Single
// outstanding" is a real design choice, not just a simplification for
// testability: it's what makes "never issues overlapping transactions"
// true by construction (mem_rden and mem_wren are driven by different,
// mutually exclusive FSM states) — see the module's state machine below,
// and see symbiyosys/ (step 21) for formally proving that holds across
// every reachable state, not just the paths a cocotb test happens to
// exercise.
//
// Real bug caught by cocotb (test_dma.py, first run): the read path
// originally had only one wait state between asserting mem_rden and
// sampling mem_rdata. That's one cycle too few for ANY same-clock-domain
// requester/memory pair, not just a testbench artifact: mem_rden/mem_addr
// become stable starting at the edge that asserts them (call it E0), a
// standard synchronous memory samples them and registers mem_rdata
// starting at the FOLLOWING edge E1 (stable through [E1, E2)), so the
// requester can only safely sample mem_rdata at E2, not E1 -- sampling at
// E1 races the memory's own same-edge update, same failure mode as two
// independent posedge-triggered processes reading each other's not-yet-
// settled output. The original single-wait-state version only worked
// against `symbiyosys`-style formal reasoning-by-inspection because
// nobody had run it against a real registered memory yet; cocotb's
// posedge-synchronous memory model (test_dma.py's memory_model) exposed
// it immediately as a one-word lag in every copied value. Fixed by
// splitting the wait into two cycles (S_READ_WAIT1, S_READ_WAIT2) so
// there's a full extra cycle of margin before sampling.
`timescale 1ns / 1ps

module dma_controller #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32,
    parameter LEN_WIDTH   = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // Control
    input  wire                     start,
    input  wire [ADDR_WIDTH-1:0]    src_addr,
    input  wire [ADDR_WIDTH-1:0]    dst_addr,
    input  wire [LEN_WIDTH-1:0]     length,     // word count, must be >= 1
    output reg                      busy,
    output reg                      done,       // one-cycle pulse

    // Single shared synchronous memory port: one read OR one write per
    // transaction, never both — see file header.
    output reg  [ADDR_WIDTH-1:0]    mem_addr,
    output reg  [DATA_WIDTH-1:0]    mem_wdata,
    output reg                      mem_rden,
    output reg                      mem_wren,
    input  wire [DATA_WIDTH-1:0]    mem_rdata   // valid one cycle after mem_rden
);

    localparam S_IDLE       = 3'd0;
    localparam S_READ       = 3'd1;
    localparam S_READ_WAIT1 = 3'd2;
    localparam S_READ_WAIT2 = 3'd3;
    localparam S_WRITE      = 3'd4;
    localparam S_NEXT       = 3'd5;
    localparam S_DONE       = 3'd6;

    reg [2:0]            state;
    reg [ADDR_WIDTH-1:0]  cur_src;
    reg [ADDR_WIDTH-1:0]  cur_dst;
    reg [LEN_WIDTH-1:0]   remaining;
    reg [DATA_WIDTH-1:0]  word_buf;

    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            mem_addr  <= {ADDR_WIDTH{1'b0}};
            mem_wdata <= {DATA_WIDTH{1'b0}};
            mem_rden  <= 1'b0;
            mem_wren  <= 1'b0;
        end else begin
            // Defaults each cycle; only the active state overrides.
            done     <= 1'b0;
            mem_rden <= 1'b0;
            mem_wren <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        cur_src   <= src_addr;
                        cur_dst   <= dst_addr;
                        remaining <= length;
                        busy      <= 1'b1;
                        state     <= S_READ;
                    end
                end

                S_READ: begin
                    mem_addr <= cur_src;
                    mem_rden <= 1'b1;
                    state    <= S_READ_WAIT1;
                end

                S_READ_WAIT1: begin
                    state <= S_READ_WAIT2;
                end

                S_READ_WAIT2: begin
                    word_buf <= mem_rdata;
                    state    <= S_WRITE;
                end

                S_WRITE: begin
                    mem_addr  <= cur_dst;
                    mem_wdata <= word_buf;
                    mem_wren  <= 1'b1;
                    state     <= S_NEXT;
                end

                S_NEXT: begin
                    cur_src   <= cur_src + 1'b1;
                    cur_dst   <= cur_dst + 1'b1;
                    remaining <= remaining - 1'b1;
                    state     <= (remaining == {{(LEN_WIDTH-1){1'b0}}, 1'b1}) ? S_DONE : S_READ;
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
