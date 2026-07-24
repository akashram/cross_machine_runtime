# onic_shell_integration.tcl — integrate rdma_bypass.p4's compiled
# pipeline (via p4c-xsa, producing a packageable Vivado IP) into the
# OpenNIC shell's user-plugin "box" and connect it to the shell's CMAC
# (network) and DMA-engine AXI4-Stream interfaces.
#
# Like every other TCL file in fpga_engine/, this has never run against a
# real OpenNIC shell block design (no Vivado license/F1 instance locally,
# and no P4 toolchain to produce the packaged IP this script consumes as
# input -- see rdma_bypass.p4's header). The block-design command surface
# (create_bd_cell/connect_bd_intf_net against OpenNIC shell's documented
# box_250mhz/box_322mhz plugin interfaces) follows the OpenNIC shell
# project's own integration guide; treat exact interface/pin names as
# reviewed-but-unverified pending a real Vivado session with the shell's
# actual IP catalog loaded, same caveat every other unrun TCL file here
# carries.
#
# Usage (headless, no GUI), against an already-open OpenNIC shell block
# design with a user plugin box instantiated but empty:
#   vivado -mode batch -source onic_shell_integration.tcl -tclargs \
#       -bd_design <opennic_shell block design name> \
#       -p4_ip_repo <dir containing the p4c-xsa-packaged IP> \
#       -p4_ip_vlnv <vendor:library:name:version of the packaged IP> \
#       -box_inst <name to give the instantiated cell, e.g. rdma_bypass_box> \
#       -outdir <dir>

array set opts {
    -bd_design "" -p4_ip_repo "" -p4_ip_vlnv "" -box_inst "" -outdir ""
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
foreach required {-bd_design -p4_ip_repo -p4_ip_vlnv -box_inst -outdir} {
    if {$opts($required) eq ""} {
        puts "ERROR: missing required argument $required"
        exit 1
    }
}
set bd_design   $opts(-bd_design)
set p4_ip_repo  $opts(-p4_ip_repo)
set p4_ip_vlnv  $opts(-p4_ip_vlnv)
set box_inst    $opts(-box_inst)
set outdir      $opts(-outdir)

file mkdir $outdir/reports

set_property ip_repo_paths $p4_ip_repo [current_project]
update_ip_catalog

open_bd_design $bd_design

# Instantiate rdma_bypass.p4's compiled pipeline into the shell's
# already-reserved user-plugin box cell.
create_bd_cell -type ip -vlnv $p4_ip_vlnv $box_inst
puts "onic_shell_integration.tcl: instantiated $p4_ip_vlnv as $box_inst"

# OpenNIC shell's box plugin interface is AXI4-Stream in both directions
# (RX from CMAC into the box, TX from the box back toward the DMA
# engine) at the shell's configured box clock (box_250mhz for a 100G
# CMAC lane at this design's width) -- rdma_bypass.p4's ingress/egress
# ports are named to match the box's expected sub-interface names on
# that clock domain.
connect_bd_intf_net [get_bd_intf_pins $box_inst/s_axis_rx] \
    [get_bd_intf_pins cmac_usplus_0/axis_rx]
connect_bd_intf_net [get_bd_intf_pins $box_inst/m_axis_tx] \
    [get_bd_intf_pins qdma_0/axis_c2h]
connect_bd_net [get_bd_pins $box_inst/clk] [get_bd_pins box_250mhz]
connect_bd_net [get_bd_pins $box_inst/rst_n] [get_bd_pins box_250mhz_aresetn]

# AXI4-Lite control path so the host can still read pipeline counters
# (packets matched/dropped) even though the WRITE data path itself never
# touches host software -- observability, not participation in the
# per-packet decision.
connect_bd_intf_net [get_bd_intf_pins $box_inst/s_axi_ctrl] \
    [get_bd_intf_pins axi_crossbar_0/M_AXI_ctrl]

validate_bd_design
save_bd_design

report_bd_design_summary $bd_design -file $outdir/reports/bd_summary.rpt
puts "onic_shell_integration.tcl: box '$box_inst' wired to CMAC RX and QDMA TX; next -> generate_target/synth_design on $bd_design as usual"
exit 0
