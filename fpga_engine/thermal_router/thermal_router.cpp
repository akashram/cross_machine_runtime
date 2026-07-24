// thermal_router.cpp — ThermalRouter's hardware-touching half:
// read_fpga_temp_c() and fpga_allocation_fraction(). Reads the FPGA die
// temperature the same way xadc/xadc_sensors.cpp does -- XRT's
// get_info<xrt::info::device::thermal>() JSON report -- and calls into
// thermal_policy.cpp's pure allocation_fraction_for_temp() so the
// hardware path and thermal_router_sim.cpp's locally-tested path make
// the exact same decision for the same temperature, not two
// independently-written policies that could drift apart.
//
// TODO: run on F1 with XRT installed and a card enumerated. Untested --
// no XRT runtime or FPGA card available locally. Same schema-confidence
// caveat as xadc_sensors.cpp: the thermal.json field name assumed below
// ("fpga_temp") follows XRT's documented Thermal.json report shape but
// isn't confirmed against a real report until this runs.

#include "thermal_router.h"

#include <cstdio>
#include <stdexcept>
#include <string>

#include "xrt/xrt_device.h"

namespace {

// Minimal, dependency-free JSON scalar extraction -- enough to pull one
// numeric field out of get_info<>()'s report without a JSON library for
// a single value. A real deployment would more likely reuse whatever
// parser parse_xadc_json.py's Python side already validated the schema
// against, once that schema is confirmed against a real report.
float extract_temp_c(const std::string& thermal_json) {
    auto pos = thermal_json.find("\"fpga_temp\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("thermal_router: fpga_temp field not found in thermal.json");
    }
    pos = thermal_json.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("thermal_router: malformed thermal.json");
    }
    return std::stof(thermal_json.substr(pos + 1));
}

} // namespace

float ThermalRouter::read_fpga_temp_c() const {
    xrt::device device(0); // first FPGA device enumerated by XRT, same as xadc_sensors.cpp
    std::string thermal_json = device.get_info<xrt::info::device::thermal>();
    return extract_temp_c(thermal_json);
}

float ThermalRouter::fpga_allocation_fraction() const {
    float temp_c = read_fpga_temp_c();
    if (temp_c >= policy_.warning_temp_c) {
        std::printf("thermal_router: WARNING fpga die temp %.1fC\n", temp_c);
    }
    return allocation_fraction_for_temp(temp_c);
}
