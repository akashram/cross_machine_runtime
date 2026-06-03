#pragma once

// Coroutine Execution Engine — Design Doc
// =========================================================================
//
// This header implements four things that together form a coroutine runtime:
//
//   Task<T>          — a lazily-started async value backed by a C++20 coroutine
//   AwaitableEvent   — a one-shot signal a coroutine can co_await
//   AwaitableMutex   — a mutex whose lock() returns an awaitable (no thread block)
//   schedule()       — wires continuations into the WorkStealingPool
//
// Read this doc top-to-bottom before touching the code.
//
//
// =========================================================================
// 1. C++20 COROUTINE MECHANICS — what the compiler actually does
// =========================================================================
//
// A function is a coroutine when its body contains co_await, co_yield, or
// co_return.  The compiler desugars it into a state machine allocated on the
// heap (the "coroutine frame") and returns a handle to the caller immediately.
//
// Return-type protocol (promise_type)
// ------------------------------------
// For a coroutine function `Task<T> foo()`, the compiler looks up
// `Task<T>::promise_type` (or `std::coroutine_traits<Task<T>>::promise_type`).
// It then calls these methods in order:
//
//   1. promise_type p;              <- default-constructed inside the frame
//   2. Task<T> ret = p.get_return_object();   <- returned to the caller
//   3. co_await p.initial_suspend();          <- first suspension point
//      ... body executes ...
//   4. p.return_value(v);   OR  p.return_void();   <- co_return v
//      (or p.unhandled_exception() if an exception escapes)
//   5. co_await p.final_suspend() noexcept;  <- last suspension point
//
// The coroutine frame is destroyed either by:
//   - final_suspend returning a "never suspend" awaiter -> destroyed automatically
//   - final_suspend suspending -> the frame stays alive; caller calls handle.destroy()
//
// Awaiter protocol (what co_await expr does)
// -------------------------------------------
// `co_await expr` first gets an awaiter `a` from expr (via operator co_await or
// directly if expr already satisfies the awaiter concept).  Then:
//
//   if (!a.await_ready()) {
//       // save this coroutine's state
//       auto result = a.await_suspend(this_handle);
//       // depending on result type:
//       //   void   -> return to caller/resumer (suspend)
//       //   bool   -> if false, don't actually suspend (resume immediately)
//       //   coroutine_handle<> h -> "symmetric transfer": resume h instead
//       //                          (no stack growth -- tail call to h)
//   }
//   T val = a.await_resume();   // runs when this coroutine is resumed
//
// The three return types of await_suspend are the key to efficient scheduling:
//
//   void   -- simplest, but every suspend/resume pair uses two stack frames.
//   bool   -- lets the awaiter skip suspension if the result is already ready
//             (avoids suspending when an event is already set, etc.).
//   handle -- "symmetric transfer" (P0913 / Gor Nishanov).  The current
//             coroutine suspends and the returned handle is resumed as a tail
//             call by the runtime, with no additional stack frame.  This is
//             how we avoid O(depth) stack growth when chaining co_awaits.
//
// Symmetric transfer example:
//   Coroutine A co_awaits Task B.  B completes.  B's final_suspend returns
//   A's handle via symmetric transfer -> A resumes without growing the stack.
//   Without this, each level of co_await nesting adds a frame.
//
//
// =========================================================================
// 2. Task<T> DESIGN
// =========================================================================
//
// Task<T> represents an asynchronous computation that produces a T (or void).
// Constraints:
//   - Lazily started: the body does not execute until someone co_awaits the task
//     (or explicitly resumes it).  This is essential for structured concurrency:
//     the parent can wire up the continuation before the child runs.
//   - Move-only: Task owns the coroutine frame; copying is nonsensical.
//   - Result stored in the frame: the T (or exception_ptr) lives inside the
//     coroutine's promise until the awaiter reads it via await_resume().
//
// Why lazy (initial_suspend = suspend_always)?
//   Eager start (initial_suspend = suspend_never) runs the body immediately
//   during the call that creates the Task.  That means:
//     (a) the Task is live before the caller has a chance to co_await it, so
//         lifetimes are harder to reason about;
//     (b) you cannot create a Task and pass it somewhere before it begins --
//         the body has already run.
//   Lazy start defers execution to the first co_await, which is the moment the
//   caller is ready to wait.  This matches async/await in Rust and Python.
//
// final_suspend and symmetric transfer
// --------------------------------------
// When the Task's body reaches co_return v, the promise calls return_value(v)
// to stash the result, then hits final_suspend.  We must NOT destroy the frame
// here (suspend_never) because the awaiting coroutine still needs to read the
// result via await_resume().  Instead final_suspend returns a FinalAwaiter whose
// await_suspend performs symmetric transfer to the stored continuation handle:
//
//   struct FinalAwaiter {
//       bool await_ready() noexcept { return false; }
//       coroutine_handle<> await_suspend(coroutine_handle<promise_type> h) noexcept {
//           // If a continuation was registered (someone is co_awaiting this Task),
//           // resume it.  Otherwise noop_coroutine() -- frame stays suspended
//           // until Task::~Task() calls handle_.destroy().
//           auto cont = h.promise().continuation_;
//           return cont ? cont : std::noop_coroutine();
//       }
//       void await_resume() noexcept {}
//   };
//
// The coroutine frame lives until Task::~Task() calls handle_.destroy().
//
// co_await Task<T> -- the awaitable side
// ----------------------------------------
// Task<T> itself satisfies the awaitable concept.
// Its await_suspend uses symmetric transfer:
//   1. Store the caller's handle as the task's continuation_.
//   2. Return handle_ -- symmetric transfer starts the task as a tail call.
// The entire chain runs on the same thread (cache-warm, no scheduling overhead).
// See §5 for external entry points that use schedule() instead.
//
// Result storage
// ---------------
// The promise holds:  std::variant<std::monostate, T, std::exception_ptr>
//   monostate     = not yet complete
//   T             = completed with value
//   exception_ptr = completed with exception
//
// await_resume() checks the variant and either returns T or rethrows.
//
//
// =========================================================================
// 3. AwaitableEvent
// =========================================================================
//
// A one-shot binary event: starts "not set", transitions to "set" exactly once.
// Any coroutine that co_awaits an unset event suspends until set() is called.
// Any coroutine that co_awaits an already-set event passes through immediately.
//
// State machine (3 states, one atomic<void*>)
// -------------------------------------------
//   nullptr   = not set, no waiter
//   kSet      = set (sentinel (void*)1)
//   <handle>  = not set, one coroutine is waiting
//
// set():
//   old = state_.exchange(kSet, acq_rel)
//   if old is a handle: resume it (inline or via pool)
//
// await_ready(): load(acquire) == kSet  ->  skip suspension entirely.
//
// await_suspend(caller_handle):
//   CAS(nullptr -> caller_handle.address(), acq_rel/acquire)
//   success: we stored the handle, caller suspends -> return true
//   failure: state_ is already kSet (set() ran between await_ready and here)
//            -> return false: do NOT suspend
//
// The set/suspend race is resolved by the CAS: if set() already ran, the CAS
// fails, we return false, and the coroutine continues without ever suspending.
//
// Memory ordering:
//   set()           acq_rel exchange  -- release publishes writes before set()
//   await_ready()   acquire load
//   await_suspend() acq_rel CAS       -- acquire sees set()'s release on failure
//
//
// =========================================================================
// 4. AwaitableMutex
// =========================================================================
//
// A mutual-exclusion lock whose lock() returns an awaitable.  The locking
// coroutine suspends (without blocking its thread) if the mutex is held.
// The unlock path resumes the next waiter, either inline or via the pool.
//
// Design: Lewis Baker's cppcoro AsyncMutex (CppCon 2019).
//
// State encoding (one atomic<uintptr_t>)
// ----------------------------------------
//   kUnlocked         (0)  = not held, no waiters
//   kLockedNoWaiters  (1)  = held, no waiters
//   ptr to Waiter node     = held, intrusive LIFO stack of waiters
//
// Waiter nodes live on the SUSPENDED COROUTINE'S HEAP FRAME (zero alloc):
//   struct Waiter { coroutine_handle<> handle; Waiter* next; };
// A suspended coroutine's frame is heap-allocated and lives until resumed or
// destroyed, so the Waiter is valid for the full duration of the suspension.
//
// await_ready():  CAS(kUnlocked -> kLockedNoWaiters, acquire).  Fast path.
//
// await_suspend(h):
//   fill waiter_.handle = h.
//   CAS loop:
//     if state == kUnlocked: CAS -> kLockedNoWaiters, return false (don't suspend)
//     else: waiter_.next = (Waiter*)state, CAS -> &waiter_, return true (suspend)
//
// unlock():
//   acquire-load state (synchronizes-with await_suspend's release CAS so that
//   head->next is visible before we dereference it).
//   CAS loop:
//     if kLockedNoWaiters: CAS -> kUnlocked (release), done.
//     else pop head waiter, set new state = head->next or kLockedNoWaiters,
//          CAS (acq_rel), then resume/schedule head->handle.
//   "Transfer ownership": the new owner does not CAS from Unlocked -> Locked;
//   the lock is passed directly, closing the window where a third thread could
//   steal it between the unlock and the waiter's re-lock.
//
// co_await mu.lock() returns a Guard (RAII unlock on scope exit).
//
// Memory ordering:
//   lock CAS (await_ready / await_suspend success)  acquire
//   unlock CAS (kLockedNoWaiters -> kUnlocked)       release
//   waiter push CAS                                  acq_rel
//   unlock initial load + pop CAS                    acquire / acq_rel
//
//
// =========================================================================
// 5. THREAD POOL INTEGRATION
// =========================================================================
//
// schedule(pool, h) wraps h.resume() as a callable and submits it to pool.
// This is the bridge between the coroutine world and the thread pool.
//
// Where scheduling happens:
//   AwaitableEvent::set(pool)  -> schedule the waiting coroutine on pool
//   AwaitableMutex::unlock(pool) -> schedule the next waiter on pool
//   External entry points      -> call schedule(pool, task.raw_handle())
//                                 to inject a root Task into the pool
//
// Inline vs. scheduled Task start
// ---------------------------------
// co_await task uses symmetric transfer (inline), not schedule():
//   - Child runs on the same cache-warm core as the parent.
//   - No pool flooding for inner co_awaits.
//
// External entry points (tasks driven from plain threads) call schedule().
// FinalAwaiter also uses symmetric transfer (not schedule()) so continuation
// resumes on the same worker without a round-trip through the inbox.
//
//
// =========================================================================
// 6. MEMORY ORDERING SUMMARY
// =========================================================================
//
//   AwaitableEvent::set()           acq_rel exchange
//   AwaitableEvent::await_ready()   acquire load
//   AwaitableEvent::await_suspend() acq_rel/acquire CAS
//
//   AwaitableMutex lock             acquire CAS
//   AwaitableMutex waiter push      acq_rel CAS
//   AwaitableMutex unlock load      acquire
//   AwaitableMutex unlock CAS       release (no-waiters) / acq_rel (pop waiter)
//
//   Task continuation_ store        relaxed -- sequenced-before await_suspend
//                                   returns, which synchronizes-with FinalAwaiter
//   Task FinalAwaiter load          relaxed -- sequenced-after return_value()
//
// TSan note: the inline path (symmetric transfer) does not cross thread
// boundaries; TSan sees a single-thread chain.  The scheduled path goes through
// the pool's Chase-Lev release-store (submit) / acquire-load (worker pop),
// which provides the needed synchronization for data written before schedule().
//
//
// =========================================================================
// 7. LIFETIME RULES
// =========================================================================
//
//   Task<T>: owns the frame.  Destroying before co_await: frame is destroyed.
//   Destroying mid-flight (running or suspended in pool): undefined behavior.
//   Callers must ensure the Task is (a) complete, (b) co_awaited from a
//   coroutine that outlives the Task, or (c) never started.
//
//   AwaitableEvent: must outlive every coroutine that co_awaits it.
//   AwaitableMutex: must outlive all coroutines that lock it.
//   Waiter nodes: lifetime = duration of the suspension (created in
//   await_suspend, invalidated when the coroutine is resumed).
//
//
// =========================================================================
// 8. TESTING PLAN
// =========================================================================
//
//   (a) Task basics:
//       - single task co_return int, check value
//       - chain A co_awaits B co_awaits C, check all values
//       - exception in body propagates through co_await
//       - Task<void> completes without exception
//       - task destroyed before started (no crash)
//
//   (b) AwaitableEvent:
//       - already set before co_await -> no suspension
//       - set after co_await from external thread
//       - double set() does not resume twice
//
//   (c) AwaitableMutex:
//       - single acquire + release, result correct
//       - Guard RAII unlocks on scope exit
//       - two coroutines contend: one suspends, resumes after unlock
//       - N coroutines on pool increment shared counter: final == N * ITERS
//         (lost-update detection; run under TSan)
//
//   (d) Pool integration:
//       - root Task driven from plain thread via schedule()
//       - N tasks fan out on pool, latch.wait() for completion
//
//   (e) TSan: all tests run under `cmake --preset tsan`
//
// =========================================================================

#include "../ws_pool/ws_pool.h"

#include <atomic>
#include <coroutine>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace foundation {

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
template <typename T = void> class Task;
class AwaitableEvent;
class AwaitableMutex;

// ---------------------------------------------------------------------------
// schedule() -- submit a coroutine handle to a WorkStealingPool
// ---------------------------------------------------------------------------
inline void schedule(WorkStealingPool& pool, std::coroutine_handle<> h) {
    pool.submit([h]() mutable { h.resume(); });
}

// ---------------------------------------------------------------------------
// Task<T> -- lazily-started coroutine returning T
// ---------------------------------------------------------------------------

template <typename T>
class Task {
public:
    struct promise_type {
        Task get_return_object() noexcept {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        struct FinalAwaiter {
            bool await_ready() noexcept { return false; }

            std::coroutine_handle<> await_suspend(
                    std::coroutine_handle<promise_type> h) noexcept {
                // Symmetric transfer to the awaiting coroutine.
                // If no one is awaiting (root task), go to noop so the frame
                // stays suspended until Task::~Task() destroys it.
                auto cont = h.promise().continuation_;
                return cont ? cont : std::noop_coroutine();
            }

            void await_resume() noexcept {}
        };

        FinalAwaiter final_suspend() noexcept { return {}; }

        void return_value(T v) {
            result_.template emplace<1>(std::move(v));
        }

        void unhandled_exception() {
            result_.template emplace<2>(std::current_exception());
        }

        std::coroutine_handle<> continuation_{};
        std::variant<std::monostate, T, std::exception_ptr> result_;
    };

    explicit Task(std::coroutine_handle<promise_type> h) noexcept : handle_(h) {}
    Task(Task&& o) noexcept : handle_(std::exchange(o.handle_, {})) {}
    Task& operator=(Task&& o) noexcept {
        if (this != &o) {
            if (handle_) handle_.destroy();
            handle_ = std::exchange(o.handle_, {});
        }
        return *this;
    }
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task() { if (handle_) handle_.destroy(); }

    // Awaitable interface -----------------------------------------------------

    bool await_ready() noexcept { return false; }

    // Symmetric transfer: store caller as continuation, tail-call into the task.
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> caller) noexcept {
        handle_.promise().continuation_ = caller;
        return handle_;
    }

    T await_resume() {
        auto& r = handle_.promise().result_;
        if (std::holds_alternative<std::exception_ptr>(r))
            std::rethrow_exception(std::get<std::exception_ptr>(r));
        return std::get<T>(std::move(r));
    }

    // raw_handle() is intentionally public: used by test helpers (sync_wait,
    // schedule-on-pool lambdas) that need to start a root Task from outside a
    // coroutine context.
    std::coroutine_handle<promise_type> raw_handle() const noexcept { return handle_; }

private:
    std::coroutine_handle<promise_type> handle_;
};

// ---------------------------------------------------------------------------
// Task<void> specialisation
// ---------------------------------------------------------------------------
template <>
class Task<void> {
public:
    struct promise_type {
        Task get_return_object() noexcept {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        struct FinalAwaiter {
            bool await_ready() noexcept { return false; }

            std::coroutine_handle<> await_suspend(
                    std::coroutine_handle<promise_type> h) noexcept {
                auto cont = h.promise().continuation_;
                return cont ? cont : std::noop_coroutine();
            }

            void await_resume() noexcept {}
        };

        FinalAwaiter final_suspend() noexcept { return {}; }

        void return_void() {
            result_.emplace<1>(std::monostate{});
        }

        void unhandled_exception() {
            result_.emplace<2>(std::current_exception());
        }

        std::coroutine_handle<> continuation_{};
        std::variant<std::monostate, std::monostate, std::exception_ptr> result_;
    };

    explicit Task(std::coroutine_handle<promise_type> h) noexcept : handle_(h) {}
    Task(Task&& o) noexcept : handle_(std::exchange(o.handle_, {})) {}
    Task& operator=(Task&& o) noexcept {
        if (this != &o) {
            if (handle_) handle_.destroy();
            handle_ = std::exchange(o.handle_, {});
        }
        return *this;
    }
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task() { if (handle_) handle_.destroy(); }

    bool await_ready() noexcept { return false; }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> caller) noexcept {
        handle_.promise().continuation_ = caller;
        return handle_;
    }

    void await_resume() {
        auto& r = handle_.promise().result_;
        if (std::holds_alternative<std::exception_ptr>(r))
            std::rethrow_exception(std::get<std::exception_ptr>(r));
    }

    std::coroutine_handle<promise_type> raw_handle() const noexcept { return handle_; }

private:
    std::coroutine_handle<promise_type> handle_;
};

// ---------------------------------------------------------------------------
// AwaitableEvent -- one-shot signal
// ---------------------------------------------------------------------------

class AwaitableEvent {
public:
    AwaitableEvent() noexcept : state_(kNotSet) {}

    AwaitableEvent(const AwaitableEvent&) = delete;
    AwaitableEvent& operator=(const AwaitableEvent&) = delete;

    // Signal the event.  If a coroutine is waiting, resume or schedule it.
    void set(WorkStealingPool* pool = nullptr) noexcept {
        uintptr_t old = state_.exchange(kSet, std::memory_order_acq_rel);
        if (old != kNotSet && old != kSet) {
            auto h = std::coroutine_handle<>::from_address(
                reinterpret_cast<void*>(old));
            if (pool)
                schedule(*pool, h);
            else
                h.resume();
        }
    }

    struct Awaiter {
        explicit Awaiter(AwaitableEvent& e, WorkStealingPool* pool) noexcept
            : event_(e), pool_(pool) {}

        bool await_ready() noexcept {
            return event_.state_.load(std::memory_order_acquire) == kSet;
        }

        // Returns false (don't suspend) if the event was set between await_ready
        // and here -- the CAS will fail and we see kSet.
        bool await_suspend(std::coroutine_handle<> h) noexcept {
            uintptr_t expected = kNotSet;
            return event_.state_.compare_exchange_strong(
                expected,
                reinterpret_cast<uintptr_t>(h.address()),
                std::memory_order_acq_rel,
                std::memory_order_acquire);
        }

        void await_resume() noexcept {}

        AwaitableEvent& event_;
        WorkStealingPool* pool_;
    };

    Awaiter operator co_await() noexcept { return Awaiter{*this, nullptr}; }
    Awaiter on(WorkStealingPool& pool) noexcept { return Awaiter{*this, &pool}; }

private:
    static constexpr uintptr_t kNotSet = 0;
    static constexpr uintptr_t kSet    = 1;
    std::atomic<uintptr_t> state_;
};

// ---------------------------------------------------------------------------
// AwaitableMutex -- async mutual exclusion
// ---------------------------------------------------------------------------

class AwaitableMutex {
public:
    AwaitableMutex() noexcept : state_(kUnlocked) {}

    AwaitableMutex(const AwaitableMutex&) = delete;
    AwaitableMutex& operator=(const AwaitableMutex&) = delete;

    // RAII guard: destructor calls unlock().
    class Guard {
    public:
        explicit Guard(AwaitableMutex& m, WorkStealingPool* pool) noexcept
            : mutex_(&m), pool_(pool) {}
        Guard(Guard&& o) noexcept
            : mutex_(std::exchange(o.mutex_, nullptr)), pool_(o.pool_) {}
        Guard(const Guard&) = delete;
        ~Guard() { if (mutex_) mutex_->unlock(pool_); }
    private:
        AwaitableMutex* mutex_;
        WorkStealingPool* pool_;
    };

    struct LockAwaiter {
        explicit LockAwaiter(AwaitableMutex& m, WorkStealingPool* pool) noexcept
            : mutex_(m), pool_(pool) {}

        // Fast path: try to acquire without suspending.
        bool await_ready() noexcept {
            uintptr_t expected = kUnlocked;
            return mutex_.state_.compare_exchange_strong(
                expected, kLockedNoWaiters,
                std::memory_order_acquire,
                std::memory_order_relaxed);
        }

        // Slow path: push onto the waiter stack or acquire if it just unlocked.
        bool await_suspend(std::coroutine_handle<> h) noexcept {
            waiter_.handle = h;
            uintptr_t s = mutex_.state_.load(std::memory_order_relaxed);
            while (true) {
                if (s == kUnlocked) {
                    // Raced with unlock(); try to acquire directly.
                    if (mutex_.state_.compare_exchange_weak(
                            s, kLockedNoWaiters,
                            std::memory_order_acquire,
                            std::memory_order_relaxed))
                        return false;  // acquired, don't suspend
                    continue;
                }
                // Lock is held: push this waiter onto the stack.
                // release-store on waiter_.next pairs with the acquire-load in
                // unlock() to give TSan an explicit HB edge on this field.
                waiter_.next.store(reinterpret_cast<Waiter*>(s),
                                   std::memory_order_release);
                if (mutex_.state_.compare_exchange_weak(
                        s, reinterpret_cast<uintptr_t>(&waiter_),
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed))
                    return true;  // in wait list, suspend
                // CAS failed: s reloaded, retry
            }
        }

        // Returns the RAII guard so `auto g = co_await mu.lock()` works.
        Guard await_resume() noexcept { return Guard{mutex_, pool_}; }

        AwaitableMutex& mutex_;
        WorkStealingPool* pool_;

        // next is atomic to give TSan an explicit HB edge between the
        // release-store in await_suspend and the acquire-load in unlock().
        // Without this, TSan is confused by the ABA pattern on state_: the
        // same &waiter_ address appears in state_ in consecutive iterations,
        // and TSan may associate unlock()'s acquire with the OLD push rather
        // than the current one, making it unable to see the HB from
        // Waiter::Waiter() through await_suspend's CAS to unlock()'s read.
        struct Waiter {
            std::coroutine_handle<> handle;
            std::atomic<Waiter*> next{nullptr};
        };
        Waiter waiter_;
    };

    LockAwaiter lock(WorkStealingPool* pool = nullptr) noexcept {
        return LockAwaiter{*this, pool};
    }

    // Release the mutex.  If there are waiters, transfer lock ownership and
    // resume the head waiter (inline or via pool).
    void unlock(WorkStealingPool* pool = nullptr) noexcept {
        // Acquire so that head->next (written in await_suspend with release)
        // is visible before we dereference it below.
        uintptr_t s = state_.load(std::memory_order_acquire);
        while (true) {
            if (s == kLockedNoWaiters) {
                if (state_.compare_exchange_weak(
                        s, kUnlocked,
                        std::memory_order_release,
                        std::memory_order_relaxed))
                    return;
                continue;
            }
            // s is a Waiter*: pop head, transfer lock ownership.
            auto* head = reinterpret_cast<LockAwaiter::Waiter*>(s);
            // acquire-load pairs with the release-store in await_suspend,
            // giving TSan the direct HB edge it needs on this field.
            auto* next_ptr = head->next.load(std::memory_order_acquire);
            uintptr_t next_state = next_ptr
                ? reinterpret_cast<uintptr_t>(next_ptr)
                : kLockedNoWaiters;
            // acq_rel: acquire synchronizes with the waiter that pushed head;
            // release publishes the lock transfer to the resumed coroutine.
            if (state_.compare_exchange_weak(
                    s, next_state,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                if (pool)
                    schedule(*pool, head->handle);
                else
                    head->handle.resume();
                return;
            }
            // CAS failed (failure acquire reloads s): retry
        }
    }

private:
    static constexpr uintptr_t kUnlocked        = 0;
    static constexpr uintptr_t kLockedNoWaiters = 1;

    // 0   = unlocked
    // 1   = locked, no waiters
    // ptr = locked, intrusive LIFO stack of LockAwaiter::Waiter nodes
    std::atomic<uintptr_t> state_;
};

} // namespace foundation
