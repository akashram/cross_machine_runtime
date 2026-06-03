#pragma once

// NUMA-Aware Allocator
// =========================================================================
//
// NUMA (Non-Uniform Memory Access) is the dominant memory architecture on
// multi-socket servers.  A 2-socket Intel Xeon system has two memory
// controllers — one per socket — and two sets of DIMMs.  A core on socket 0
// accessing its local DIMMs sees ~80 ns latency; the same core accessing
// socket 1's DIMMs sees ~140 ns (the "remote" penalty, ~1.7x overhead).
// On 4-socket systems the worst-case hop count is larger still.
//
// The Linux kernel models this as NUMA nodes: a node is a set of CPUs that
// share a local memory controller.
//
//
// Key concepts
// ------------
// First-touch policy (Linux default):
//   A virtual page is assigned to a DRAM bank the first time it is written.
//   The assignment goes to the NUMA node of the CPU that triggered the fault.
//   If thread A (node 0) initialises a buffer and thread B (node 1) reads it
//   later, every cache miss is a remote access — even if B "owns" the data
//   logically.  Explicit node binding (mbind) avoids this.
//
// mbind(2) — Linux syscall:
//   Sets NUMA memory policy for a virtual address range.  MPOL_BIND with a
//   node mask forces future page faults in that range to use the target node's
//   DRAM.  We call it immediately after mmap, before any page fault occurs, so
//   MPOL_MF_MOVE (which migrates already-faulted pages) is not needed.
//
// sched_getcpu(3) — Linux glibc extension:
//   Returns the CPU index the calling thread is currently on.  Usually a
//   vsyscall (no ring transition).  Used to look up the calling thread's NUMA
//   node from the topology table.
//
// pthread_setaffinity_np(3) — Linux glibc extension:
//   Hard-pins a thread to a CPU set.  The "np" suffix = "non-portable" —
//   POSIX does not standardise CPU affinity.  Once pinned the kernel will not
//   migrate the thread, preventing cross-node "ping-pong".
//
// macOS reality:
//   macOS targets single-socket consumer hardware; it has no NUMA nodes,
//   no mbind(), no sched_getcpu(), no hard CPU affinity.  The Mach thread
//   affinity API (thread_policy_set / THREAD_AFFINITY_POLICY) is advisory.
//   All Linux-specific paths are compiled out on macOS; the allocator
//   degrades to a single-node arena identical to ThreadLocalArena (step 14).
//
//
// Linux topology detection (no libnuma required)
// -----------------------------------------------
// We read sysfs directly instead of linking libnuma:
//   /sys/devices/system/node/node{N}/cpulist — CPU list for node N
//     e.g. "0-11,24-35" = CPUs 0..11 and 24..35
//
// libnuma is a thin wrapper around the same sysfs paths and syscalls used
// here.  Implementing it directly exposes the mechanism and avoids a library
// dependency.
//
//
// When to use what
// ----------------
//   ThreadLocalArena   Per-thread, zero contention, no NUMA awareness.
//                      Best when alloc and access both happen on the same
//                      thread (common case).
//
//   NumaArena          Explicitly node-bound slabs.  Use when:
//                      • Data is init'd by one thread, consumed by another
//                        thread on the same node.
//                      • Threads are pinned to nodes; you want DRAM-local
//                        allocations regardless of who first-touches.
//                      • Profiling shows cross-node cache miss overhead.
//
//
// Expected cross-NUMA penalty (2-socket Intel Xeon, indicative):
//   Local DRAM alloc        ~10-15 ns
//   Remote DRAM alloc       ~18-25 ns   (+50-100%)
//   Local cache miss        ~80  ns
//   Remote cache miss       ~140 ns     (+75%)
//   Varies by CPU, UPI/QPI link width, and BIOS settings.
//
// =========================================================================

#include "../arena/arena.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#ifdef __linux__
#  include <pthread.h>
#  include <sched.h>        // sched_getcpu
#  include <sys/syscall.h>  // SYS_mbind
#endif
#ifdef __APPLE__
#  include <sys/sysctl.h>
#  include <mach/mach.h>
#endif

namespace foundation {

// =========================================================================
// detail helpers
// =========================================================================
namespace detail {

// Parse "0-3,8,10-12" → list of CPU IDs.
inline std::vector<int> parse_cpulist(const char* s) noexcept {
    std::vector<int> out;
    while (*s && *s != '\n') {
        int a = 0;
        while (*s >= '0' && *s <= '9') a = a * 10 + (*s++ - '0');
        int b = a;
        if (*s == '-') { ++s; b = 0; while (*s >= '0' && *s <= '9') b = b*10 + (*s++ - '0'); }
        for (int c = a; c <= b; ++c) out.push_back(c);
        if (*s == ',') ++s;
    }
    return out;
}

// mmap a slab and bind it to a NUMA node (Linux).  On macOS: plain mmap.
// Called before any page fault, so MPOL_MF_MOVE is not needed.
inline void* numa_mmap(int node, std::size_t size) noexcept {
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;

#ifdef __linux__
    // MPOL_BIND = 2 (from <numaif.h>, replicated to avoid library dep)
    static constexpr int kMpolBind = 2;
    unsigned long mask = 1UL << static_cast<unsigned>(node < 64 ? node : 0);
    // maxnode = 64: covers any realistic server (up to 64 NUMA nodes)
    ::syscall(SYS_mbind, p, size, kMpolBind, &mask, 64UL, 0UL);
    // Ignore errors: slab still works, just without binding.
    // This occurs when CONFIG_NUMA is not set or the node doesn't exist.
#else
    (void)node;
#endif

    return p;
}

} // namespace detail

// =========================================================================
// NumaTopology — immutable snapshot of NUMA topology
// =========================================================================
struct NumaTopology {
    int                           num_nodes{1};
    int                           num_cpus{1};
    std::vector<int>              cpu_to_node;    // [cpu_id] → node_id
    std::vector<std::vector<int>> node_cpus;      // [node_id] → [cpu_id, ...]

    static NumaTopology detect() noexcept {
#ifdef __linux__
        return detect_linux();
#else
        return detect_single_node();
#endif
    }

    // Which NUMA node is the calling thread currently on?
    int current_node() const noexcept {
#ifdef __linux__
        int cpu = ::sched_getcpu();
        if (cpu >= 0 && cpu < static_cast<int>(cpu_to_node.size()))
            return cpu_to_node[cpu];
#endif
        return 0;
    }

private:
#ifdef __linux__
    static NumaTopology detect_linux() noexcept {
        NumaTopology t;
        t.num_cpus = static_cast<int>(::sysconf(_SC_NPROCESSORS_CONF));
        t.cpu_to_node.assign(t.num_cpus, 0);

        for (int n = 0; n < 256; ++n) {
            char path[128];
            std::snprintf(path, sizeof(path),
                "/sys/devices/system/node/node%d/cpulist", n);
            FILE* f = std::fopen(path, "r");
            if (!f) break;

            char buf[512] = {};
            if (std::fgets(buf, sizeof(buf), f)) {
                auto cpus = detail::parse_cpulist(buf);
                t.node_cpus.push_back(cpus);
                for (int c : cpus)
                    if (c < t.num_cpus)
                        t.cpu_to_node[c] = n;
            }
            std::fclose(f);
        }

        t.num_nodes = static_cast<int>(t.node_cpus.size());
        if (t.num_nodes == 0) return detect_single_node();  // CONFIG_NUMA off
        return t;
    }
#endif

    static NumaTopology detect_single_node() noexcept {
        NumaTopology t;
        t.num_nodes = 1;
#if defined(__APPLE__)
        std::size_t len = sizeof(t.num_cpus);
        ::sysctlbyname("hw.logicalcpu", &t.num_cpus, &len, nullptr, 0);
#elif defined(__linux__)
        t.num_cpus = static_cast<int>(::sysconf(_SC_NPROCESSORS_CONF));
#endif
        t.cpu_to_node.assign(static_cast<std::size_t>(t.num_cpus), 0);
        std::vector<int> all(static_cast<std::size_t>(t.num_cpus));
        for (int i = 0; i < t.num_cpus; ++i) all[static_cast<std::size_t>(i)] = i;
        t.node_cpus.push_back(std::move(all));
        return t;
    }
};

// =========================================================================
// bind_thread_to_node — pin calling thread to a node's CPU set
// =========================================================================
// Linux: hard binding via pthread_setaffinity_np.
// macOS: single-node only; returns true since node 0 is the only option.
inline bool bind_thread_to_node(int node, const NumaTopology& topo) noexcept {
    if (node < 0 || node >= topo.num_nodes) return false;

#if defined(__linux__)
    cpu_set_t cs;
    CPU_ZERO(&cs);
    for (int c : topo.node_cpus[node]) CPU_SET(c, &cs);
    return ::pthread_setaffinity_np(::pthread_self(), sizeof(cs), &cs) == 0;
#else
    // macOS: advisory-only; thread_policy_set with THREAD_AFFINITY_POLICY
    // is a scheduling hint the OS may ignore.  For a single-node system,
    // all CPUs are local — nothing to bind.
    return node == 0;
#endif
}

// =========================================================================
// NumaArena — one SizeClassedArena per NUMA node
// =========================================================================
//
// Construction via NumaArena::make() (factory, not constructor) so we can
// return a partially-valid object when some nodes fail to allocate.
//
// Each arena's slab is created with detail::numa_mmap(node, size), which
// calls mmap + mbind(MPOL_BIND) on Linux.  Page faults on the slab will
// be served from that node's DRAM.
//
// Thread safety: each per-node SizeClassedArena is single-threaded.  The
// expected usage pattern is that only threads pinned to node N access
// arena N.  If cross-node access is needed, guard with an external lock.

class NumaArena {
public:
    static constexpr std::size_t kDefaultSlabPerNode = 64u << 20;  // 64 MiB

    // Factory: detect topology, create per-node arenas.
    static NumaArena make(
        std::size_t slab_per_node = kDefaultSlabPerNode) noexcept
    {
        return make(NumaTopology::detect(), slab_per_node);
    }

    static NumaArena make(
        const NumaTopology& topo,
        std::size_t slab_per_node = kDefaultSlabPerNode) noexcept
    {
        NumaArena na(topo);
        na.arenas_.reserve(static_cast<std::size_t>(topo.num_nodes));
        for (int n = 0; n < topo.num_nodes; ++n)
            na.arenas_.push_back(make_node_arena(n, slab_per_node));
        return na;
    }

    NumaArena(NumaArena&&)            = default;
    NumaArena& operator=(NumaArena&&) = default;
    NumaArena(const NumaArena&)       = delete;
    NumaArena& operator=(const NumaArena&) = delete;

    // Allocate `size` bytes from node `node`'s arena.
    void* alloc_on_node(int node, std::size_t size) noexcept {
        if (!valid_node(node)) return nullptr;
        return arenas_[static_cast<std::size_t>(node)]->alloc(size);
    }

    // Free `ptr` (originally alloc_on_node(node, size)) back to its arena.
    void free_on_node(int node, void* ptr, std::size_t size) noexcept {
        if (!valid_node(node)) return;
        arenas_[static_cast<std::size_t>(node)]->free(ptr, size);
    }

    // Allocate from whichever node the calling thread is currently on.
    void* alloc_local(std::size_t size) noexcept {
        return alloc_on_node(topo_.current_node(), size);
    }

    void free_local(void* ptr, std::size_t size) noexcept {
        free_on_node(topo_.current_node(), ptr, size);
    }

    void reset_node(int node, bool release_pages = false) noexcept {
        if (!valid_node(node)) return;
        arenas_[static_cast<std::size_t>(node)]->reset(release_pages);
    }

    bool valid_node(int n) const noexcept {
        auto u = static_cast<std::size_t>(n);
        return n >= 0 && u < arenas_.size() && arenas_[u];
    }
    const NumaTopology& topology() const noexcept { return topo_; }

private:
    explicit NumaArena(const NumaTopology& topo) noexcept : topo_(topo) {}

    // Create a SizeClassedArena and bind its slab to NUMA node `node`.
    // The arena mmaps its own slab; we then call mbind on that slab.
    // Since no pages have faulted in yet, future faults will use node's DRAM.
    static std::unique_ptr<SizeClassedArena>
    make_node_arena(int node, std::size_t size) noexcept {
        auto sa = std::make_unique<SizeClassedArena>(size);
        if (!sa->arena().ok()) return nullptr;

#ifdef __linux__
        static constexpr int kMpolBind = 2;
        unsigned long mask = 1UL << static_cast<unsigned>(node < 64 ? node : 0);
        ::syscall(SYS_mbind,
                  sa->arena().base(), sa->arena().capacity(),
                  kMpolBind, &mask, 64UL, 0UL);
#else
        (void)node;
#endif
        return sa;
    }

    NumaTopology topo_;
    std::vector<std::unique_ptr<SizeClassedArena>> arenas_;
};

} // namespace foundation
