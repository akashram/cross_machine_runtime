// xadc_sensors.cpp — read FPGA die temperature and voltage rails from the
// host via XRT's device sensor API.
//
// PLAN.md step 18 asks to read die temperature and voltage rails
// programmatically from host and integrate into the thermal-aware router
// (thermal_router/thermal_router.h, step 25). On AWS F1 the physical XADC
// (SYSMON on UltraScale+) lives inside the static shell, not the user's
// programmable region — a user kernel has no AXI-Lite path to it. XRT
// already reads it on the host's behalf (the same sensors `xbutil examine
// -r thermal` / `-r electrical` prints) and exposes the result through
// xrt::device::get_info<>(), so that's the real API surface here, not raw
// register access — same reasoning as dma_orchestration.cpp using XRT's
// native C++ API instead of hand-rolled PCIe BAR access.
//
// TODO: run on F1 with XRT installed and a card enumerated. Untested — no
// XRT runtime or FPGA card available locally. The get_info<> query kinds
// and field names below follow XRT's documented sensor report schemas
// (the same Thermal.json / Electrical.json shapes `xbutil examine` renders
// from); parse_xadc_json.py's self-test exercises that schema without
// needing hardware, but the exact field set XRT emits on a given shell
// version isn't confirmed until this runs for real.
#include <cstdio>
#include <fstream>
#include <string>

#include "xrt/xrt_device.h"

namespace {

// Writes a get_info<> JSON blob to disk so parse_xadc_json.py can turn it
// into structured readings without linking against XRT itself — same
// split as power_ci (report_power.tcl produces the .rpt, parse_power_report.py
// parses it) and clock_gating (power_gating.tcl vs power_delta.py).
void dump_json(const std::string& label, const std::string& json, const std::string& out_path) {
    std::ofstream out(out_path);
    out << json;
    std::printf("xadc_sensors: wrote %s report (%zu bytes) -> %s\n",
                label.c_str(), json.size(), out_path.c_str());
}

} // namespace

int main(int argc, char** argv) {
    std::string outdir = argc > 1 ? argv[1] : ".";

    xrt::device device(0); // first FPGA device enumerated by XRT

    // Both queries return the sensor report as a JSON string (the same
    // payload xbutil examine's -f json output embeds for these two report
    // kinds); see file header for schema-confidence caveat.
    std::string thermal_json = device.get_info<xrt::info::device::thermal>();
    std::string electrical_json = device.get_info<xrt::info::device::electrical>();

    dump_json("thermal", thermal_json, outdir + "/thermal.json");
    dump_json("electrical", electrical_json, outdir + "/electrical.json");

    std::printf("xadc_sensors: next -> python3 parse_xadc_json.py %s/thermal.json %s/electrical.json\n",
                outdir.c_str(), outdir.c_str());
    return 0;
}
