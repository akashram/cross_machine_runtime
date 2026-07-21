# Step 6: Autograd engine — portable, header-only (matrix.h, autograd.h,
# mlp.h), no CUDA/Linux dependency. The shared foundation steps 7-25 build
# on: reverse-mode tape over a small dense Matrix, plus a toy MLP model.

add_library(Autograd INTERFACE)
target_include_directories(Autograd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(autograd_test autograd_test.cpp)
target_link_libraries(autograd_test PRIVATE Autograd)

enable_testing()
add_test(NAME autograd_test COMMAND autograd_test)
