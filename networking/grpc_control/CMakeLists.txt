# Step 6: gRPC + protobuf control plane. Needs protobuf's and gRPC's
# codegen (protoc --cpp_out / --grpc_out) to turn scheduler.proto into
# scheduler.pb.h/.cc and scheduler.grpc.pb.h/.cc before
# scheduler_service.cpp's #include "scheduler.grpc.pb.h" resolves — so
# this whole step is gated behind both packages being found, not just
# Linux (gRPC is available via brew on macOS too, but not assumed
# installed here).

find_package(Protobuf QUIET)
find_package(gRPC QUIET)

if(NOT Protobuf_FOUND OR NOT gRPC_FOUND)
  message(STATUS "grpc_control: protobuf and/or gRPC not found — skipped")
  return()
endif()

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)

add_custom_command(
  OUTPUT ${GENERATED_DIR}/scheduler.pb.cc ${GENERATED_DIR}/scheduler.pb.h
         ${GENERATED_DIR}/scheduler.grpc.pb.cc ${GENERATED_DIR}/scheduler.grpc.pb.h
  COMMAND protobuf::protoc
  ARGS --cpp_out=${GENERATED_DIR} --grpc_out=${GENERATED_DIR}
       --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
       -I ${CMAKE_CURRENT_SOURCE_DIR} ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
)

add_library(SchedulerProto
  ${GENERATED_DIR}/scheduler.pb.cc
  ${GENERATED_DIR}/scheduler.grpc.pb.cc
)
target_include_directories(SchedulerProto PUBLIC ${GENERATED_DIR})
target_link_libraries(SchedulerProto PUBLIC protobuf::libprotobuf gRPC::grpc++)

add_library(SchedulerService scheduler_service.cpp)
target_link_libraries(SchedulerService PUBLIC SchedulerProto)

add_executable(scheduler_server scheduler_server.cpp)
target_link_libraries(scheduler_server PRIVATE SchedulerService gRPC::grpc++)
