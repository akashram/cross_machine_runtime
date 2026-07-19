# Step 5: PTP clock synchronization — Linux only (linux/ptp_clock.h,
# /dev/ptp* hardware clocks).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(PtpSync ptp_sync.cpp)
  target_include_directories(PtpSync PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()
