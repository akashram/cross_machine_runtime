// scheduler_server.cpp — standalone gRPC server hosting SchedulerServiceImpl.
// Usage: scheduler_server [listen_addr:port]

#include "scheduler_service.h"

#include <grpcpp/grpcpp.h>

#include <cstdio>

int main(int argc, char **argv) {
  std::string addr = argc > 1 ? argv[1] : "0.0.0.0:50051";

  runtime::SchedulerServiceImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::printf("scheduler_server listening on %s\n", addr.c_str());
  server->Wait();
  return 0;
}
