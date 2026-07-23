# synth.tcl — Vivado headless synthesis + implementation + bitstream pipeline
#
# Generic, parameterized build for any kernel in fpga_engine/. Every later
# step (dot_product, loop_opt, ml_kernel, ...) reuses this script instead of
# hand-rolling its own Vivado flow, so the synth/impl/bitstream/report
# sequence and its failure handling live in exactly one place.
#
# TODO: run on F1 with Vivado 2022.x installed. Untestable off Linux + a
# licensed Vivado install, so this has never executed end-to-end.
#
# Usage (headless, no GUI, e.g. from CI or run_pipeline.sh):
#   vivado -mode batch -source synth.tcl -tclargs \
#       -top <top_module> \
#       -ip_dir <dir with HLS-exported .xci/.zip IP> \
#       -xdc <constraints file> \
#       -outdir <output directory>
#
# Exit code is 0 only if synthesis, implementation, and bitstream generation
# all completed AND timing closed (worst negative slack >= 0). CI should
# treat any non-zero exit as a build failure, not just a missing artifact.

set part xcvu9p-flgb2104-2-i

# ---- parse -tclargs ---------------------------------------------------
array set opts {-top "" -ip_dir "" -xdc "" -outdir ""}
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
foreach required {-top -ip_dir -xdc -outdir} {
    if {$opts($required) eq ""} {
        puts "ERROR: missing required argument $required"
        exit 1
    }
}
set top       $opts(-top)
set ip_dir    $opts(-ip_dir)
set xdc_file  $opts(-xdc)
set outdir    $opts(-outdir)

file mkdir $outdir
file mkdir $outdir/reports

# ---- project setup -----------------------------------------------------
create_project -force build_project $outdir/build_project -part $part

# HLS export produces either loose .xci or packaged IP-XACT .zip; pick up
# whichever this kernel's export step produced.
foreach ip_file [glob -nocomplain -directory $ip_dir *.xci] {
    read_ip $ip_file
}
foreach ip_zip [glob -nocomplain -directory $ip_dir *.zip] {
    set_property ip_repo_paths $ip_dir [current_project]
    update_ip_catalog
}

read_xdc $xdc_file

# ---- synthesis -----------------------------------------------------------
synth_design -top $top -part $part
write_checkpoint -force $outdir/post_synth.dcp
report_utilization -file $outdir/reports/utilization_synth.rpt

# ---- implementation -------------------------------------------------------
opt_design
place_design
report_utilization -file $outdir/reports/utilization_placed.rpt
route_design
write_checkpoint -force $outdir/post_route.dcp

# ---- bitstream -------------------------------------------------------------
write_bitstream -force $outdir/${top}.bit

# ---- reports (always generated, regardless of timing outcome, so CI can
#      archive them even on a failing build) ---------------------------------
report_timing_summary -file $outdir/reports/timing.rpt
report_utilization    -file $outdir/reports/utilization.rpt
report_power          -file $outdir/reports/power.rpt
report_drc            -file $outdir/reports/drc.rpt

# ---- timing closure gate ---------------------------------------------------
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "synth.tcl: worst negative slack (setup) = $wns ns"
if {$wns < 0} {
    puts "synth.tcl: FAILED — timing not closed (WNS $wns ns < 0)"
    exit 1
}

puts "synth.tcl: build of '$top' complete — bitstream at $outdir/${top}.bit"
exit 0
