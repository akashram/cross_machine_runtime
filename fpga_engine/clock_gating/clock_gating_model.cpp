// clock_gating_model.cpp — duty-cycle-driven dynamic power reduction model.
//
// PLAN.md step 17 asks to measure dynamic power reduction from clock
// gating with Vivado power analysis. There's no F1 instance here (see
// power_gating.tcl), so this file does the part that doesn't need one: a
// first-order model of how much dynamic power gating actually saves as a
// function of how often the gated logic is doing useful work (duty
// cycle), and where the break-even point is once the gating logic's own
// overhead (BUFGCE + enable-tree) is accounted for.
//
// Clock gating (Vivado's `-gated_clock_conversion`, see power_gating.tcl)
// promotes a common register-bank enable into a real gated clock net, so
// registers stop toggling -- and stop burning dynamic power -- during
// cycles where they wouldn't have changed value anyway. `gated_mlp.cpp`
// gives ml_kernel_mlp's registers exactly that kind of common enable (a
// `valid` input gating every write), so this model is the piece that
// says whether adding it is actually worth it for a given traffic
// pattern before spending F1 time measuring it.
//
// Same caveat as critical_path_model.cpp / slr_crossing_model.cpp:
// kGatingOverheadFrac is a first-order engineering approximation (BUFGCE
// + enable-tree power as a fraction of the ungated design's dynamic
// power), not a datasheet quote -- the real number depends on how much
// logic shares the gated enable and how it's placed. What doesn't depend
// on getting that constant exactly right: the model's structural claim
// that reduction is *negative* at duty cycle -> 100% (gating a signal
// that's always active is pure overhead, never a win) and grows toward
// (1 - overhead) as duty cycle -> 0.

#include <cstdio>
#include <vector>

namespace {

// BUFGCE + enable-tree overhead as a fraction of the ungated design's
// dynamic power -- see file header caveat.
constexpr double kGatingOverheadFrac = 0.02;

// Modeled dynamic power with gating, normalized to ungated dynamic power
// = 1.0: gated registers only toggle (draw dynamic power) during the
// duty_cycle fraction of time they're actually updated, plus the fixed
// overhead of the gating logic itself, which runs every cycle regardless.
double gated_power_normalized(double duty_cycle) {
    return duty_cycle + kGatingOverheadFrac;
}

double reduction_pct(double duty_cycle) {
    return (1.0 - gated_power_normalized(duty_cycle)) * 100.0;
}

} // namespace

int main() {
    const std::vector<double> duty_cycles = {1.00, 0.50, 0.25, 0.10, 0.05, 0.01};

    std::printf("=== dynamic power reduction vs. duty cycle (gating overhead = %.0f%%) ===\n",
                kGatingOverheadFrac * 100.0);
    for (double duty : duty_cycles) {
        double reduction = reduction_pct(duty);
        std::printf("duty=%5.1f%% | gated/ungated dynamic power = %.3f | reduction = %+6.1f%% %s\n",
                    duty * 100.0, gated_power_normalized(duty), reduction,
                    reduction < 0.0 ? "(NET LOSS — do not gate)" : "");
    }

    double breakeven_duty = 1.0 - kGatingOverheadFrac;
    std::printf(
        "\nbreak-even duty cycle = %.1f%% — gate only if the design's expected duty"
        " cycle is below this; above it, gating overhead exceeds its own savings.\n",
        breakeven_duty * 100.0);

    std::printf("\n=== applied to gated_mlp.cpp (ml_kernel_mlp + valid-gated registers) ===\n");
    std::printf(
        "Bursty inference (valid asserted ~1/10 cycles, duty=10%%): predicted reduction = %.1f%%"
        " — gating is worth adding.\n",
        reduction_pct(0.10));
    std::printf(
        "Saturated streaming (valid asserted every cycle, duty=100%%): predicted reduction = %+.1f%%"
        " — gating is a net loss; do not add it to a design run this way.\n",
        reduction_pct(1.00));

    return 0;
}
