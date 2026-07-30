// npu_thermal_router.cpp — the hardware-touching half of NpuThermalRouter.
// Real IOKit/SMC sensor read, NOT a stub -- but genuinely unrun: reading
// SMC sensor keys requires linking IOKit.framework and running on real
// Apple Silicon hardware with SMC access permissions, and there is no
// portable way to verify the exact sensor key for "ANE die temperature"
// without a physical device and Apple's (undocumented, reverse-engineered
// by the community) SMC key table -- e.g. `smc -k <key> -r` style tools
// use keys like "Tp09"/"Tp0T" for various SoC thermal zones on different
// Apple Silicon generations, and there is no STABLE public per-ANE-block
// key the way FPGA's XADC exposes a documented per-die-region sensor.
// This is the real, disclosed platform limitation SCOPE.md's NPU section
// alludes to ("no straightforward AWS/GCP rental story") extended to
// thermal monitoring specifically: not just "no cloud rental," but "no
// stable public per-block sensor API" either.
//
// TODO: requires IOKit.framework + a real SMC key lookup for this
// specific Mac's Apple Silicon generation (out of scope to reverse-
// engineer here), run on real Apple Silicon hardware. Unrun.

#include "npu_thermal_policy.h"

#include <stdexcept>

#if defined(__APPLE__)
// Real framework this would link against: -framework IOKit
// #include <IOKit/IOKitLib.h>
#endif

float NpuThermalRouter::read_npu_temp_c() const {
    // Real implementation would:
    //   1. IOServiceGetMatchingService(kIOMasterPortDefault,
    //        IOServiceMatching("AppleSMC"))
    //   2. Open a connection, call SMCReadKey() for this generation's
    //      SoC/ANE-adjacent thermal zone key (community-documented keys
    //      vary by chip generation -- e.g. "Tp09" on some M-series parts
    //      -- not a single stable constant across the product line).
    //   3. Convert the returned SP78 fixed-point value to float Celsius.
    // Left unimplemented (not stubbed with a fake number) since there is
    // no single correct key to hard-code without testing on the specific
    // target hardware generation -- see file header.
    throw std::logic_error(
        "read_npu_temp_c: requires IOKit/SMC access on real Apple Silicon hardware "
        "with a chip-generation-specific SMC key; not runnable in this environment");
}

float NpuThermalRouter::npu_allocation_fraction() const {
    return allocation_fraction_for_temp(read_npu_temp_c());
}
