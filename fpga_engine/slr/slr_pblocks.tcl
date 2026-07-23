# slr_pblocks.tcl — constrain kernel instances to specific SLRs, measure
# the crossing-penalty impact of doing so vs. leaving placement unconstrained.
#
# Deliberately does NOT hardcode SLICE/DSP/clock-region coordinate ranges
# for VU9P's 3 SLRs -- getting that floorplan wrong from outside Vivado
# (no access to the real part database here) would silently constrain the
# wrong area or nothing at all. get_slrs and SLR_INDEX are real, documented
# Vivado commands/properties for SSI (stacked silicon, multi-die) parts;
# this script queries the loaded part for its actual SLRs and builds
# pblocks from what the tool reports, rather than guessing. Like every
# other TCL file in fpga_engine/, this has never run against a real
# checkpoint (no Vivado license/F1 instance locally) -- treat exact flag
# names (report_design_analysis's SLR-crossing-nets option in particular)
# as reviewed-but-unverified pending a real Vivado session.
#
# Usage (headless, no GUI), against a post-synthesis checkpoint before
# opt_design/place_design have run:
#   vivado -mode batch -source slr_pblocks.tcl -tclargs \
#       -checkpoint <post_synth.dcp> \
#       -pin_pattern <cell name glob to confine to a single SLR, e.g. "*ml_kernel_mlp*"> \
#       -outdir <dir>
#
# What this measures: builds two variants from the same checkpoint --
# unconstrained (whatever SLR(s) the default placer spreads -pin_pattern's
# cells across) and pinned (all of -pin_pattern's cells confined to
# pblock_slr0) -- and reports the per-SLR utilization and any SLR-crossing
# nets for both, so the crossing count/WNS difference between the two is
# directly comparable.

array set opts {-checkpoint "" -pin_pattern "" -outdir ""}
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
set checkpoint   $opts(-checkpoint)
set pin_pattern  $opts(-pin_pattern)
set outdir       $opts(-outdir)

file mkdir $outdir/reports

# ---- variant 1: unconstrained baseline --------------------------------
open_checkpoint $checkpoint
opt_design
place_design
route_design
report_utilization -slr -file $outdir/reports/utilization_per_slr_unconstrained.rpt
report_design_analysis -slr_crossing_nets -file $outdir/reports/slr_crossings_unconstrained.rpt
set wns_unconstrained [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "slr_pblocks.tcl: unconstrained WNS = $wns_unconstrained ns"
close_design

# ---- variant 2: pin -pin_pattern's cells to a single SLR --------------
open_checkpoint $checkpoint

set slrs [get_slrs]
puts "slr_pblocks.tcl: part has [llength $slrs] SLR(s): $slrs"
if {[llength $slrs] < 2} {
    puts "slr_pblocks.tcl: single-SLR part or SLRs not detected -- nothing to constrain, exiting"
    exit 0
}

foreach slr $slrs {
    set idx [get_property SLR_INDEX $slr]
    set pblock_name "pblock_slr${idx}"
    create_pblock $pblock_name
    resize_pblock [get_pblocks $pblock_name] -add $slr
    puts "slr_pblocks.tcl: created $pblock_name over $slr"
}

if {$pin_pattern ne ""} {
    set target_cells [get_cells -hierarchical -filter "NAME =~ \"$pin_pattern\""]
    if {[llength $target_cells] > 0} {
        add_cells_to_pblock [get_pblocks pblock_slr0] $target_cells
        # CONTAIN_ROUTING forces the tool to actually honor the pblock
        # boundary during routing, not just placement -- without it a
        # pblock is only a placement hint and routed nets can still leave
        # the SLR, which would defeat the point of pinning in the first
        # place.
        set_property CONTAIN_ROUTING true [get_pblocks pblock_slr0]
        puts "slr_pblocks.tcl: pinned [llength $target_cells] cells matching '$pin_pattern' to pblock_slr0"
    } else {
        puts "slr_pblocks.tcl: WARNING no cells matched pattern '$pin_pattern'"
    }
}

opt_design
place_design
route_design
report_utilization -slr -file $outdir/reports/utilization_per_slr_pinned.rpt
report_design_analysis -slr_crossing_nets -file $outdir/reports/slr_crossings_pinned.rpt
set wns_pinned [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "slr_pblocks.tcl: pinned WNS = $wns_pinned ns"
write_checkpoint -force $outdir/post_slr_pinned.dcp

puts "slr_pblocks.tcl: WNS unconstrained=$wns_unconstrained ns, pinned=$wns_pinned ns"
puts "slr_pblocks.tcl: compare reports/slr_crossings_unconstrained.rpt vs. reports/slr_crossings_pinned.rpt for the crossing-count delta"
exit 0
