# Step 3: SLA-aware (EDF) scheduler — portable, builds and runs
# everywhere including this Mac.

add_library(SlaScheduler sla_scheduler.cpp)
target_include_directories(SlaScheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(SlaScheduler PUBLIC cxx_std_23)
set_project_warnings(SlaScheduler)

add_executable(sla_scheduler_test sla_scheduler_test.cpp)
target_link_libraries(sla_scheduler_test PRIVATE SlaScheduler)
set_project_warnings(sla_scheduler_test)

add_executable(sla_scheduler_bench sla_scheduler_bench.cpp)
target_link_libraries(sla_scheduler_bench PRIVATE SlaScheduler)
set_project_warnings(sla_scheduler_bench)

enable_testing()
add_test(NAME sla_scheduler_test COMMAND sla_scheduler_test)
