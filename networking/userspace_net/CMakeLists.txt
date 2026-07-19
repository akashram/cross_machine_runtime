# Step 9: userspace networking stack — built on af_xdp (step 8), same
# Linux + libbpf requirement.
if(TARGET AfXdpSocket)
  add_library(UserspaceNetStack net_stack.cpp)
  target_link_libraries(UserspaceNetStack PUBLIC AfXdpSocket)
  target_include_directories(UserspaceNetStack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/networking/af_xdp
  )
else()
  message(STATUS "userspace_net: af_xdp target unavailable — skipped")
endif()
