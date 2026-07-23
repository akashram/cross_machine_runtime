# power_gating.tcl — build gated_mlp.cpp with and without gated clock
# conversion, at a specified valid-signal duty cycle, and report_power both.
#
# Real Vivado commands throughout: synth_design's -gated_clock_conversion
# option promotes a register bank's common write-enable (gated_mlp.cpp's
# `valid`, see that file's header) into an actual gated clock net;
# set_switching_activity overrides the power estimator's assumed toggle
# rate/static probability for a signal when no real SAIF capture exists,
# which is exactly the case here -- this lets report_power's estimate
# reflect the duty cycle clock_gating_model.cpp's prediction is stated
# for, rather than Vivado's default (usually pessimistic, ~50%) activity
# assumption. Like every other TCL file in fpga_engine/, this has never
# run against a real design (no Vivado license/F1 instance locally) --
# treat -gated_clock_conversion's exact accepted values and
# set_switching_activity's flags as reviewed-but-unverified pending a
# real Vivado session.
#
# Usage (headless, no GUI):
#   vivado -mode batch -source power_gating.tcl -tclargs \
#       -top gated_mlp \
#       -ip_dir <dir with HLS-exported IP> \
#       -xdc <constraints file> \
#       -duty_cycle 0.10 \
#       -outdir <dir>

set part xcvu9p-flgb2104-2-i

array set opts {-top "" -ip_dir "" -xdc "" -duty_cycle "0.5" -outdir ""}
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
set top        $opts(-top)
set ip_dir     $opts(-ip_dir)
set xdc_file   $opts(-xdc)
set duty_cycle $opts(-duty_cycle)
set outdir     $opts(-outdir)

file mkdir $outdir
file mkdir $outdir/reports

proc build_and_report_power {variant_name gated_clock_conversion ip_dir xdc_file top part outdir duty_cycle} {
    create_project -force build_$variant_name $outdir/build_$variant_name -part $part

    foreach ip_file [glob -nocomplain -directory $ip_dir *.xci] { read_ip $ip_file }
    foreach ip_zip [glob -nocomplain -directory $ip_dir *.zip] {
        set_property ip_repo_paths $ip_dir [current_project]
        update_ip_catalog
    }
    read_xdc $xdc_file

    if {$gated_clock_conversion} {
        synth_design -top $top -part $part -gated_clock_conversion on
    } else {
        synth_design -top $top -part $part
    }
    opt_design
    place_design
    route_design

    # Annotate the valid port's activity to match the traffic pattern
    # clock_gating_model.cpp's prediction is for, since no real SAIF
    # capture exists yet -- without this, report_power falls back to a
    # generic default activity estimate that wouldn't reflect any
    # particular duty cycle.
    set_switching_activity -static_probability $duty_cycle \
        -toggle_rate [expr {$duty_cycle * 100.0}] [get_ports valid]

    report_power -file $outdir/reports/power_${variant_name}.rpt
    write_checkpoint -force $outdir/post_route_${variant_name}.dcp
    close_design
}

puts "power_gating.tcl: building '$top' ungated (baseline) at duty_cycle=$duty_cycle"
build_and_report_power "ungated" 0 $ip_dir $xdc_file $top $part $outdir $duty_cycle

puts "power_gating.tcl: building '$top' with -gated_clock_conversion on, duty_cycle=$duty_cycle"
build_and_report_power "gated" 1 $ip_dir $xdc_file $top $part $outdir $duty_cycle

puts "power_gating.tcl: done — compare reports/power_ungated.rpt vs. reports/power_gated.rpt"
puts "power_gating.tcl: run power_delta.py on both to compute the measured reduction"
exit 0
