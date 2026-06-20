#pragma once
#include <cstddef>
// TODO: implement on Linux with libfabric + EFA

class RdmaEndpoint {
public:
    // Initialize libfabric endpoint with EFA provider
    bool init(const char* prov_name = "efa");
    // Connect to remote endpoint
    bool connect(const char* remote_addr, uint16_t port);
    // Two-sided send (blocking)
    ssize_t send(const void* buf, size_t len);
    // Two-sided recv (blocking)
    ssize_t recv(void* buf, size_t len);
    void close();
};
