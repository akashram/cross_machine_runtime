# Step 1: PagedAttention-style KV block allocator/table — portable, builds
# and runs everywhere including this Mac. See paged_kv_cache.h's header
# comment for why this owns bookkeeping only, not device K/V bytes.

add_library(PagedKV paged_kv_cache.cpp)
target_include_directories(PagedKV PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(PagedKV PUBLIC cxx_std_23)
set_project_warnings(PagedKV)

add_executable(paged_kv_test paged_kv_test.cpp)
target_link_libraries(paged_kv_test PRIVATE PagedKV)
target_include_directories(paged_kv_test PRIVATE ${CMAKE_SOURCE_DIR}/foundation)
set_project_warnings(paged_kv_test)

enable_testing()
add_test(NAME paged_kv_test COMMAND paged_kv_test)
