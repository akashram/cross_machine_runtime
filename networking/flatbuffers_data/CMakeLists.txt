# Step 7: FlatBuffers data plane. Needs flatc to turn tensor.fbs into
# tensor_generated.h before tensor_codec.cpp's #include resolves.

find_package(Flatbuffers QUIET)
if(NOT Flatbuffers_FOUND)
  message(STATUS "flatbuffers_data: FlatBuffers not found — skipped")
  return()
endif()

set(FBS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor.fbs)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

add_custom_command(
  OUTPUT ${GENERATED_DIR}/tensor_generated.h
  COMMAND flatbuffers::flatc
  ARGS --cpp -o ${GENERATED_DIR} ${FBS_FILE}
  DEPENDS ${FBS_FILE}
)
add_custom_target(TensorFbsGen DEPENDS ${GENERATED_DIR}/tensor_generated.h)

add_library(TensorCodec tensor_codec.cpp)
add_dependencies(TensorCodec TensorFbsGen)
target_include_directories(TensorCodec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${GENERATED_DIR})
target_link_libraries(TensorCodec PUBLIC flatbuffers::flatbuffers)

add_executable(tensor_codec_bench tensor_codec_bench.cpp)
target_link_libraries(tensor_codec_bench PRIVATE TensorCodec)
