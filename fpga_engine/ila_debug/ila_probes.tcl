# ila_probes.tcl — insert an ILA debug core on axi_stream/axi_passthrough's
# AXI4-Stream ports, re-implement with the debug core, program the device,
# and capture a trace to CSV.
#
# Real Vivado debug-core-insertion + Hardware Manager commands throughout
# (UG908's post-synthesis debug flow, headless/Tcl form): create_debug_core
# instantiates an ILA IP, connect_debug_port wires each probe to a net
# from the post-synthesis netlist, implement_debug_core re-runs
# opt/place/route with the ILA folded in, and the open_hw_manager block
# programs the bitstream over JTAG/PCIe and triggers a capture. Same
# caveat as every other TCL file in fpga_engine/: never run against a real
# design (no Vivado license/F1 instance locally) — treat exact debug-core
# property names and the hw_ila command sequence as reviewed against
# UG908/UG835 but unverified pending a real Vivado session.
#
# write_hw_ila_data's CSV column layout is Vivado's own (per-probe radix
# columns, a header block before the data rows) — NOT the same schema
# axi_trace_checker.py reads. Converting one to the other needs a real
# capture to look at first, so it's left as a TODO rather than guessed at
# here; axi_trace_checker.py's canonical schema (cycle,tvalid,tready,tdata,tlast)
# is documented in its own header.
#
# Usage (headless, no GUI):
#   vivado -mode batch -source ila_probes.tcl -tclargs \
#       -top axi_passthrough \
#       -ip_dir <dir with HLS-exported IP> \
#       -xdc <constraints file> \
#       -outdir <output directory> \
#       -hw_server localhost:3121

set part xcvu9p-flgb2104-2-i

array set opts {-top "" -ip_dir "" -xdc "" -outdir "" -hw_server "localhost:3121"}
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
set hw_server $opts(-hw_server)

file mkdir $outdir
file mkdir $outdir/reports

# ---- synthesize (same shape as tcl_pipeline/synth.tcl) ---------------------
create_project -force build_ila $outdir/build_ila -part $part
foreach ip_file [glob -nocomplain -directory $ip_dir *.xci] { read_ip $ip_file }
foreach ip_zip [glob -nocomplain -directory $ip_dir *.zip] {
    set_property ip_repo_paths $ip_dir [current_project]
    update_ip_catalog
}
read_xdc $xdc_file
synth_design -top $top -part $part
write_checkpoint -force $outdir/post_synth.dcp

# ---- insert ILA on the AXI4-Stream ports -----------------------------------
# axi_passthrough's ports are hls::stream<ap_axiu<32,1,1,1>> named
# in_stream/out_stream; Vitis HLS expands each into TDATA/TVALID/TREADY/
# TLAST nets named <port>_TDATA, <port>_TVALID, etc. Probe both sides of
# the passthrough so a stall or corruption can be localized to the input
# handshake vs. the output handshake.
set probe_nets {
    in_stream_TVALID  in_stream_TREADY  in_stream_TDATA  in_stream_TLAST
    out_stream_TVALID out_stream_TREADY out_stream_TDATA out_stream_TLAST
}
foreach net $probe_nets {
    if {[llength [get_nets -quiet $net]] == 0} {
        puts "WARNING: net '$net' not found post-synthesis — HLS may have named it differently; check post_synth.dcp with report_property before trusting the probe list"
    }
    set_property MARK_DEBUG true [get_nets -quiet $net]
}

create_debug_core u_ila_axi ila
set_property C_DATA_DEPTH 4096 [get_debug_cores u_ila_axi]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_axi]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_axi]
set_property C_INPUT_PIPE_STAGES 0 [get_debug_cores u_ila_axi]
set_property ALL_PROBE_SAME_MU true [get_debug_cores u_ila_axi]
connect_debug_port u_ila_axi/clk [get_nets ap_clk]

set probe_idx 0
foreach net $probe_nets {
    if {$probe_idx > 0} { create_debug_port u_ila_axi probe }
    set port_width [expr {[string match "*TDATA" $net] ? 32 : 1}]
    set_property port_width $port_width [get_debug_ports u_ila_axi/probe$probe_idx]
    connect_debug_port u_ila_axi/probe$probe_idx [get_nets $net]
    incr probe_idx
}

implement_debug_core

# ---- re-implement with the ILA folded in -----------------------------------
opt_design
place_design
route_design
write_checkpoint -force $outdir/post_route_debug.dcp
write_debug_probes -force $outdir/debug_probes.ltx
write_bitstream -force $outdir/${top}_debug.bit

# ---- program the device and capture a trace --------------------------------
# Trigger on in_stream_TVALID rising so the capture starts at the first
# beat of a transfer rather than an arbitrary idle cycle.
open_hw_manager
connect_hw_server -url $hw_server
open_hw_target
set device [lindex [get_hw_devices xcvu9p*] 0]
current_hw_device $device
refresh_hw_device -update_hw_probes false $device
set_property PROBES.FILE $outdir/debug_probes.ltx $device
set_property FULL_PROBES.FILE $outdir/debug_probes.ltx $device
set_property PROGRAM.FILE $outdir/${top}_debug.bit $device
program_hw_devices $device
refresh_hw_device $device

set ila_core [get_hw_ilas -of_objects $device]
set_property TRIGGER_COMPARE_VALUE eq1'b1 [get_hw_probes */in_stream_TVALID -of_objects $ila_core]
run_hw_ila $ila_core
wait_on_hw_ila $ila_core -timeout 30

set capture [upload_hw_ila_data $ila_core]
write_hw_ila_data -csv_file $outdir/reports/ila_capture_raw.csv $capture

puts "ila_probes.tcl: raw capture at $outdir/reports/ila_capture_raw.csv"
puts "ila_probes.tcl: reformat to axi_trace_checker.py's schema (see this file's header), then:"
puts "ila_probes.tcl:   python3 axi_trace_checker.py <reformatted>.csv"
exit 0
