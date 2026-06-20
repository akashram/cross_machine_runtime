#pragma once
#include <string>
#include <vector>
#include <functional>
// TODO: implement on Linux

enum class RaftState { FOLLOWER, CANDIDATE, LEADER };

struct LogEntry {
    uint64_t term;
    uint64_t index;
    std::string command;
};

class RaftNode {
public:
    RaftNode(int id, const std::vector<std::string>& peer_addrs);

    void start();
    void stop();

    // Propose a command. Blocks until committed or returns false on timeout.
    bool propose(const std::string& command, uint64_t timeout_ms = 5000);

    RaftState state() const;
    int       leader_id() const;

    // Register callback for committed entries
    void on_commit(std::function<void(const LogEntry&)> cb);

private:
    int  id_;
    // TODO: terms, log, election timer, peer connections
};
