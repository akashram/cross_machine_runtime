# gRPC + Protobuf Control Plane

**Status: code-complete, not yet built — requires gRPC + protobuf installed
(Linux target deployment; the dependency itself isn't Linux-only, just not
assumed installed here).**

## What this measures
gRPC service for scheduler RPC: node registration, health checks, task dispatch.
Measures RPC latency (p50/p99) and throughput (RPCs/sec).

## Design
`SchedulerServiceImpl` (`scheduler_service.cpp`) is in-memory, no
persistence or leader election — durability and replication of this
control-plane state is Raft's job (step 19), layered underneath rather
than reimplemented here. `WaitTask` blocks on a condition variable until
`completeTask()` fires, rather than polling — a polling implementation
would make the p50/p99 RPC latency benchmark measure polling-interval
jitter instead of real dispatch-to-completion latency. `CMakeLists.txt`
drives `protoc --cpp_out --grpc_out` directly (not `protobuf_generate`'s
CMake helper) so the generated-file paths are explicit and cacheable; the
whole step no-ops if `find_package(Protobuf)`/`find_package(gRPC)` fail,
independent of the Linux gate the rest of `networking/` sits behind.

## Results
TODO: run on Linux with gRPC.

| RPC | Latency p50 (µs) | Latency p99 (µs) | Throughput (req/s) |
|-----|-----------------|-----------------|-------------------|
| RegisterNode | TODO | TODO | TODO |
| HealthCheck | TODO | TODO | TODO |
| DispatchTask | TODO | TODO | TODO |

## Hardware notes
- Required: Linux, gRPC and protobuf installed
