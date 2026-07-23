# close_timing.tcl — post-route timing closure pass
#
# synth.tcl (tcl_pipeline/) gates every build on WNS >= 0 but does nothing
# to fix a failure beyond reporting it. This script is what runs against a
# checkpoint that failed that gate: it identifies the top-10 worst setup
# paths, gets Vivado's own QoR-suggestion analysis on what to do about
# them, applies the general-purpose fixes (retiming, extra placement
# effort), and re-checks WNS. Every command below is a real Vivado Tcl
# command (report_timing, report_high_fanout_nets, report_qor_suggestions,
# phys_opt_design, place_design, route_design) -- this has never run
# against a real checkpoint (no Vivado license/F1 instance locally), so
# treat the fix sequencing as reviewed-but-unverified, same caveat as
# every other TCL file in fpga_engine/.
#
# Usage (headless, no GUI):
#   vivado -mode batch -source close_timing.tcl -tclargs \
#       -checkpoint <post_route.dcp that failed synth.tcl's WNS gate> \
#       -outdir <dir to write reports/updated checkpoint into>

array set opts {-checkpoint "" -outdir ""}
for {set i 0} {$i < [llength $argv]} {incr i 2} {
    set key [lindex $argv $i]
    set val [lindex $argv [expr {$i + 1}]]
    if {[info exists opts($key)]} {
        set opts($key) $val
    } else {
        puts "ERROR: unknown argument $key"
        exit 1
    }
}
foreach required {-checkpoint -outdir} {
    if {$opts($required) eq ""} {
        puts "ERROR: missing required argument $required"
        exit 1
    }
}
set checkpoint $opts(-checkpoint)
set outdir     $opts(-outdir)

file mkdir $outdir/reports

open_checkpoint $checkpoint

# ---- step 1: identify the top-10 worst setup paths and why -----------
report_timing -max_paths 10 -nworst 1 -delay_type max -setup \
    -sort_by slack -file $outdir/reports/top10_critical_paths.rpt

report_high_fanout_nets -input_pins -timing \
    -file $outdir/reports/high_fanout_nets.rpt

# report_qor_suggestions runs Vivado's own analysis of the implemented
# design and proposes concrete directives/constraint changes -- the real,
# documented mechanism for "here is what to try next" instead of manually
# eyeballing the raw path reports above.
report_qor_suggestions -file $outdir/reports/qor_suggestions.rpt
write_qor_suggestions -force $outdir/reports/qor_suggestions.rqs

set wns_before [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "close_timing.tcl: WNS before fixes = $wns_before ns"

# ---- step 2: retiming pass ---------------------------------------------
# -retime moves existing registers across combinational boundaries to
# balance stage delay without changing latency or functional behavior --
# the placement-independent fix, and the first one to try since it's the
# cheapest (no re-place/route needed if it alone closes timing).
phys_opt_design -retime -directive AggressiveExplore

set wns_after_retime [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "close_timing.tcl: WNS after phys_opt_design -retime = $wns_after_retime ns"

# ---- step 3: placement pass, only if retiming alone wasn't enough -----
# A path still failing after -retime is more likely route-dominated
# (interconnect delay from congestion or long physical distance, e.g. an
# SLR crossing -- see the slr/ step) than logic-dominated, so the fix here
# is placement effort, not more logic restructuring: apply Vivado's own
# suggested constraints/directives from qor_suggestions, then re-place
# with extra net-delay weighting and re-route.
if {$wns_after_retime < 0} {
    read_qor_suggestions $outdir/reports/qor_suggestions.rqs
    place_design -directive ExtraNetDelay_high -post_place_opt
    phys_opt_design -directive AggressiveExplore
    route_design -directive Explore
}

set wns_final [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "close_timing.tcl: WNS final = $wns_final ns"

write_checkpoint -force $outdir/post_timing_closure.dcp
report_timing_summary -file $outdir/reports/timing_final.rpt

if {$wns_final < 0} {
    puts "close_timing.tcl: FAILED — still not closed (WNS $wns_final ns < 0). See qor_suggestions.rpt for remaining recommendations."
    exit 1
}

puts "close_timing.tcl: timing closed (WNS $wns_final ns >= 0)"
exit 0
