cmake_minimum_required(VERSION 3.25)
project(cross_machine_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/Sanitizers.cmake)
include(cmake/CompilerWarnings.cmake)

enable_testing()

add_subdirectory(foundation)
