# Phase 16 gap fix: a real process-per-rank driver for
# k8s/training/statefulset.yaml -- reuses the same Autograd/ZeRO-1/grad-
# clipping/checkpoint headers full_training_loop/training_loop_test.cpp
# does, over a real TcpChannel per OS process instead of threads sharing
# one process. Portable (no CUDA/Linux dependency), same as
# full_training_loop itself.

find_package(Threads REQUIRED)

add_executable(training_worker training_worker.cpp)
target_include_directories(training_worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(training_worker PRIVATE Autograd Collectives RingAllreduce Threads::Threads)
