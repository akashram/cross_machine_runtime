# Step 3: unified observability CLI dashboard — portable, builds and
# runs everywhere including this Mac.

add_library(Dashboard dashboard.cpp)
target_include_directories(Dashboard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(Dashboard PUBLIC cxx_std_23)
set_project_warnings(Dashboard)

add_executable(dashboard_report dashboard_main.cpp)
target_link_libraries(dashboard_report PRIVATE Dashboard)
set_project_warnings(dashboard_report)

add_executable(dashboard_test dashboard_test.cpp)
target_link_libraries(dashboard_test PRIVATE Dashboard)
set_project_warnings(dashboard_test)

enable_testing()
add_test(NAME dashboard_test COMMAND dashboard_test)
