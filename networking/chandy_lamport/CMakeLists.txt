# Step 18: Chandy-Lamport distributed snapshots — portable, builds and
# runs everywhere including this Mac.

add_library(ChandyLamport chandy_lamport.cpp)
target_link_libraries(ChandyLamport PUBLIC NetCommon)
target_include_directories(ChandyLamport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(chandy_lamport_test chandy_lamport_test.cpp)
target_link_libraries(chandy_lamport_test PRIVATE ChandyLamport)

enable_testing()
add_test(NAME chandy_lamport_test COMMAND chandy_lamport_test)
