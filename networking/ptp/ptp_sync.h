#pragma once
#include <cstdint>
// Requires Linux + a PTP-capable NIC exposing a /dev/ptp* hardware clock
// (ENA on AWS supports this) + ptp4l running to actually discipline that
// clock against a master. See ptp_sync.cpp for the PTP_SYS_OFFSET ioctl
// this uses to read it.

struct ClockSyncStats {
    double offset_mean_ns;
    double offset_stddev_ns;
    double max_error_ns;
};

// Query current offset between the system clock (CLOCK_REALTIME) and the
// NIC's PTP hardware clock (PHC), in nanoseconds, via PTP_SYS_OFFSET.
// This is the offset *within this host* between CLOCK_REALTIME and the
// PHC — ptp4l is what disciplines CLOCK_REALTIME (or the PHC, in
// hardware-timestamping mode) against the network's PTP master; this
// function reads whatever residual offset that discipline loop leaves.
int64_t measure_offset_ns(const char* ptp_device = "/dev/ptp0");

// Measure synchronization accuracy over N samples (mean/stddev/max of the
// offset above), sampled once per millisecond.
ClockSyncStats measure_accuracy(int n_samples, const char* ptp_device = "/dev/ptp0");
