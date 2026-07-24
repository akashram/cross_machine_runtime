"""test_axi.py — cocotb testbench for axi_stream_passthrough.v.

Runs against the hand-written RTL model of axi_stream/axi_passthrough.cpp
(see that file's header for why this is a separate hand-written module
rather than HLS-generated RTL). Two tests:

  1. test_axi_passthrough_full_throughput — no backpressure anywhere;
     confirms data integrity, TLAST alignment, and II=1 (one word
     accepted per cycle, matching the HLS kernel's PIPELINE II=1 target
     from axi_stream/dot_product's sibling steps).
  2. test_axi_passthrough_backpressure — randomized stalls on both the
     source (s_axis_tvalid) and sink (m_axis_tready) sides, wrapped in
     with_timeout so a real protocol deadlock fails the test instead of
     hanging the simulator forever. This is the "handshake always
     resolves" property ila_debug/README.md's step describes checking by
     eye on a real capture; here it's checked by running the actual RTL
     against a random stall pattern instead.

Run: `make SIM=icarus` (see Makefile).
"""
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, with_timeout
from cocotb.utils import get_sim_time


async def reset_dut(dut):
    dut.aresetn.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 0
    for _ in range(4):
        await RisingEdge(dut.aclk)
    dut.aresetn.value = 1
    await RisingEdge(dut.aclk)


class AxiStreamSource:
    """Drives s_axis_t*. `valid_pattern`, if given, is an infinite
    generator of 0/1 consulted before offering each word (models a
    source that itself isn't always ready to produce data, not just
    downstream backpressure)."""

    def __init__(self, dut, valid_pattern=None):
        self.dut = dut
        self.valid_pattern = valid_pattern

    async def send(self, words):
        for data, last in words:
            if self.valid_pattern is not None:
                while not next(self.valid_pattern):
                    self.dut.s_axis_tvalid.value = 0
                    await RisingEdge(self.dut.aclk)

            self.dut.s_axis_tdata.value = data
            self.dut.s_axis_tlast.value = int(last)
            self.dut.s_axis_tvalid.value = 1

            while True:
                await RisingEdge(self.dut.aclk)
                if int(self.dut.s_axis_tready.value) == 1:
                    break
        self.dut.s_axis_tvalid.value = 0
        self.dut.s_axis_tlast.value = 0


class AxiStreamSink:
    """Drives m_axis_tready per `ready_pattern` (infinite generator of
    0/1, default always-ready) and collects (data, last) for every
    accepted word."""

    def __init__(self, dut, ready_pattern=None):
        self.dut = dut
        self.ready_pattern = ready_pattern
        self.received = []

    async def run_until(self, n_words):
        while len(self.received) < n_words:
            ready = 1 if self.ready_pattern is None else next(self.ready_pattern)
            self.dut.m_axis_tready.value = ready
            await RisingEdge(self.dut.aclk)
            if int(self.dut.m_axis_tvalid.value) == 1 and ready == 1:
                data = int(self.dut.m_axis_tdata.value)
                last = int(self.dut.m_axis_tlast.value)
                self.received.append((data, last))


def _always(value):
    while True:
        yield value


def _random_bits(seed, p_high):
    rng = random.Random(seed)
    while True:
        yield 1 if rng.random() < p_high else 0


@cocotb.test()
async def test_axi_passthrough_full_throughput(dut):
    cocotb.start_soon(Clock(dut.aclk, 2, unit="ns").start())
    await reset_dut(dut)

    words = [(0xDEAD0000 + i, i == 9) for i in range(10)]
    source = AxiStreamSource(dut, valid_pattern=_always(1))
    sink = AxiStreamSink(dut, ready_pattern=_always(1))

    start_ns = get_sim_time(unit="ns")
    send_task = cocotb.start_soon(source.send(words))
    await with_timeout(sink.run_until(len(words)), 1000, "ns")
    await send_task
    elapsed_ns = get_sim_time(unit="ns") - start_ns

    assert sink.received == words, f"data mismatch: sent {words}, got {sink.received}"
    # 2ns period, 1-cycle register latency + 10 back-to-back transfers:
    # expect close to 10 cycles (~20-22ns), not e.g. 20 cycles (which
    # would mean II=2 instead of the target II=1).
    assert elapsed_ns <= 13 * 2, f"expected ~II=1 throughput (~20-22ns for 10 words), took {elapsed_ns}ns"


@cocotb.test()
async def test_axi_passthrough_backpressure(dut):
    cocotb.start_soon(Clock(dut.aclk, 2, unit="ns").start())
    await reset_dut(dut)

    n_words = 30
    words = [(0xB000 + i, i == n_words - 1) for i in range(n_words)]

    # Both sides stall ~40% of the time, independently and with different
    # seeds, so the run exercises every combination of source-stalled /
    # sink-stalled / both / neither.
    source = AxiStreamSource(dut, valid_pattern=_random_bits(seed=1, p_high=0.6))
    sink = AxiStreamSink(dut, ready_pattern=_random_bits(seed=2, p_high=0.6))

    send_task = cocotb.start_soon(source.send(words))
    # Generous timeout: a real deadlock (handshake that never resolves)
    # hangs until this fires and fails the test, rather than hanging the
    # simulator process forever.
    await with_timeout(sink.run_until(n_words), 5000, "ns")
    await send_task

    assert sink.received == words, f"data mismatch under backpressure: sent {words}, got {sink.received}"
