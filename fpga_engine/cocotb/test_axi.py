"""
test_axi.py — cocotb testbench for AXI4-Stream interface
TODO: run on F1 with Vivado simulation or standalone cocotb + Verilator

Validates:
- AXI4-Stream valid/ready handshake always resolves (no deadlock)
- TLAST assertion at end of packet
- Data integrity: output data == input data for passthrough kernel
"""
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge
from cocotb.clock import Clock

@cocotb.test()
async def test_axi_passthrough(dut):
    """TODO: implement on Linux with cocotb + Verilator or Vivado sim"""
    # cocotb.start_soon(Clock(dut.aclk, 2, units="ns").start())
    # await RisingEdge(dut.aclk)
    # driver = AxiStreamDriver(dut, "in_stream", dut.aclk)
    # monitor = AxiStreamMonitor(dut, "out_stream", dut.aclk)
    # await driver.send([0xDEAD, 0xBEEF, 0xCAFE])
    # received = await monitor.recv()
    # assert received == [0xDEAD, 0xBEEF, 0xCAFE]
    pass
