# Step 2: distributed tracing spans — portable, builds and runs
# everywhere including this Mac. Hand-rolled (no OTel SDK dependency —
# see tracer.h's header comment for why).

add_library(Tracer tracer.cpp)
target_include_directories(Tracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(Tracer PUBLIC cxx_std_23)
set_project_warnings(Tracer)

add_executable(tracer_test tracer_test.cpp)
target_link_libraries(tracer_test PRIVATE Tracer)
set_project_warnings(tracer_test)

enable_testing()
add_test(NAME tracer_test COMMAND tracer_test)
