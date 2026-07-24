# dfx_pblock.tcl — Dynamic Function eXchange (DFX): define a reconfigurable
# partition, implement two interface-compatible kernels as its two
# configurations (RM_A = axi_stream/axi_passthrough.cpp, RM_B =
# axi_increment.cpp -- same AXI4-Stream port interface, different body,
# see axi_increment.cpp's header), and write a full bitstream for RM_A
# plus a partial bitstream for hot-swapping to RM_B.
#
# Like every other TCL file in fpga_engine/, this has never run against a
# real checkpoint (no Vivado license/F1 instance locally) -- the command
# surface (create_pblock/HD.RECONFIGURABLE/lock_design/pr_verify) is
# UG909's documented non-project DFX flow, not guessed, but treat exact
# flag names as reviewed-but-unverified pending a real Vivado session,
# same caveat slr_pblocks.tcl and power_gating.tcl carry.
#
# Usage (headless, no GUI), against a synthesized static top that
# instantiates the reconfigurable cell as a black box, plus separately
# synthesized RM checkpoints for RM_A and RM_B:
#   vivado -mode batch -source dfx_pblock.tcl -tclargs \
#       -static_checkpoint <static_top_synth.dcp> \
#       -rm_cell <hierarchical cell name of the reconfigurable instance> \
#       -rm_a_checkpoint <axi_passthrough_synth.dcp> \
#       -rm_b_checkpoint <axi_increment_synth.dcp> \
#       -outdir <dir>
#
# What this measures: the full bitstream for configuration 1 (static +
# RM_A) and the partial bitstream for configuration 2 (static locked,
# RM_B swapped in) -- the partial bitstream's file size is what
# reconfig_time_model.cpp's prediction and pr_host_driver.cpp's measured
# load_xclbin() latency should both be compared against. pr_verify's
# report is the formal-ish check that RM_A and RM_B are actually
# interface-compatible for this partition (same boundary, same locked
# static logic) before either bitstream is trusted to hot-swap safely.

array set opts {
    -static_checkpoint "" -rm_cell "" -rm_a_checkpoint "" -rm_b_checkpoint "" -outdir ""
}
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
foreach required {-static_checkpoint -rm_cell -rm_a_checkpoint -rm_b_checkpoint -outdir} {
    if {$opts($required) eq ""} {
        puts "ERROR: missing required argument $required"
        exit 1
    }
}
set static_checkpoint $opts(-static_checkpoint)
set rm_cell            $opts(-rm_cell)
set rm_a_checkpoint    $opts(-rm_a_checkpoint)
set rm_b_checkpoint    $opts(-rm_b_checkpoint)
set outdir              $opts(-outdir)

file mkdir $outdir/reports
file mkdir $outdir/checkpoints

# ---- configuration 1: static shell + RM_A (axi_passthrough) -----------
open_checkpoint $static_checkpoint

# Load RM_A's already-synthesized netlist into the black-boxed
# reconfigurable cell before defining the pblock over it -- the pblock
# needs real cells inside it to size against.
read_checkpoint -cell $rm_cell $rm_a_checkpoint

create_pblock pblock_rm
add_cells_to_pblock [get_pblocks pblock_rm] [get_cells $rm_cell]
# HD.RECONFIGURABLE marks this pblock as a DFX reconfigurable partition
# rather than an ordinary placement hint; RESET_AFTER_RECONFIG ensures
# the region's registers come up in a known state immediately after a
# partial reconfiguration completes, before the static shell hands it
# live traffic again.
set_property HD.RECONFIGURABLE true [get_pblocks pblock_rm]
set_property RESET_AFTER_RECONFIG true [get_pblocks pblock_rm]
puts "dfx_pblock.tcl: created reconfigurable pblock over $rm_cell"

opt_design
place_design
route_design

# Lock the static logic and routing at ROUTING level: configuration 2
# below re-enters at this exact checkpoint and must not re-place/re-route
# anything outside pblock_rm, or the two configurations would stop being
# bitstream-compatible with the same static shell.
lock_design -level routing
write_checkpoint -force $outdir/checkpoints/config1_static_rm_a_routed.dcp
write_bitstream -force $outdir/config1_full.bit
puts "dfx_pblock.tcl: wrote config1 full bitstream (static + RM_A) -> $outdir/config1_full.bit"
report_utilization -file $outdir/reports/config1_utilization.rpt
close_design

# ---- configuration 2: same locked static shell + RM_B (axi_increment) -
open_checkpoint $outdir/checkpoints/config1_static_rm_a_routed.dcp
update_design -cell $rm_cell -black_box
read_checkpoint -cell $rm_cell $rm_b_checkpoint

opt_design
place_design
route_design

write_checkpoint -force $outdir/checkpoints/config2_static_rm_b_routed.dcp
# -cell restricts the bitstream write to pblock_rm's frames only -- this
# is what makes the output a PARTIAL bitstream (just the reconfigurable
# region) instead of a second full-device bitstream.
write_bitstream -force -cell [get_cells $rm_cell] $outdir/config2_rm_b_partial.bit
puts "dfx_pblock.tcl: wrote config2 partial bitstream (RM_B only) -> $outdir/config2_rm_b_partial.bit"
report_utilization -file $outdir/reports/config2_utilization.rpt
close_design

# ---- pr_verify: confirm RM_A and RM_B are safe to hot-swap ------------
# Compares the two routed configurations' static logic and reconfigurable
# partition boundary; a mismatch here (e.g. RM_B's black-box interface
# doesn't line up with what RM_A's static shell expects) means the
# partial bitstream is NOT safe to load at runtime, independent of
# whether route_design above reported success for config 2 alone.
pr_verify \
    -initial $outdir/checkpoints/config1_static_rm_a_routed.dcp \
    -additional $outdir/checkpoints/config2_static_rm_b_routed.dcp \
    -directory $outdir/reports/pr_verify

set partial_bit_size [file size $outdir/config2_rm_b_partial.bit]
puts "dfx_pblock.tcl: config2_rm_b_partial.bit = $partial_bit_size bytes"
puts "dfx_pblock.tcl: next -> reconfig_time_model.cpp predicts hot-swap time for this size;"
puts "dfx_pblock.tcl: pr_host_driver.cpp measures the real load_xclbin() latency to compare against it"
exit 0
