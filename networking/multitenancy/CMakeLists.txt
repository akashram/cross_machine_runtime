# Step 23: Multi-tenancy — portable, builds and runs everywhere
# including this Mac.

add_library(Multitenancy multitenancy.cpp)
target_include_directories(Multitenancy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(multitenancy_test multitenancy_test.cpp)
target_link_libraries(multitenancy_test PRIVATE Multitenancy)

enable_testing()
add_test(NAME multitenancy_test COMMAND multitenancy_test)
