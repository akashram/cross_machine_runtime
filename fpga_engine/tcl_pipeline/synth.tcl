# synth.tcl — Vivado headless synthesis + implementation + bitstream
# TODO: run on F1 with Vivado 2022.x installed

# set_part for VU9P on F1
set_part {xcvu9p-flgb2104-2-i}

# TODO: read all HLS-generated IP (from vitis_hls runs)
# read_ip [ glob ip/*.xci ]

# TODO: read constraints
# read_xdc constraints/top.xdc

# TODO: run synthesis
# synth_design -top top -part xcvu9p-flgb2104-2-i

# TODO: implement
# opt_design
# place_design
# route_design

# TODO: generate bitstream
# write_bitstream -force output/top.bit

# TODO: generate reports
# report_timing_summary -file reports/timing.rpt
# report_utilization    -file reports/utilization.rpt
# report_power          -file reports/power.rpt

puts "synth.tcl: STUB — run on F1 with Vivado installed"
