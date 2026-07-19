# Step 4: EFA SRD vs RC transport selection. Pure decision logic — no
# libfabric/EFA dependency — so unlike most of networking/, this builds
# and runs everywhere, including this Mac.

add_library(SrdSelector srd_selector.cpp)
target_include_directories(SrdSelector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(srd_selector_test srd_selector_test.cpp)
target_link_libraries(srd_selector_test PRIVATE SrdSelector)

enable_testing()
add_test(NAME srd_selector_test COMMAND srd_selector_test)
