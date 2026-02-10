#pragma once
#include <vector>
#include "caf/cuda/control-layer/token.hpp"
#include <chrono>


/* Meant to represent a directed acylic graph for kernel operations
 * operations or token_ptrs are meant to come in order and declared with a dependency number to indicate dependency 
 */

// status codes
#define WAITING 0
#define READY 1
#define ERROR 2

namespace caf::cuda {

class kernel_graph {
public:
    using clock_t = std::chrono::steady_clock;

    // for caf's messaging system do not use
    kernel_graph() = default;

    kernel_graph(int device_number,
                 int stream_id,
                 int dependency_number)
        : device_number_(device_number),
          stream_id_(stream_id),
          dependency_number_(dependency_number),
          last_move_(clock_t::now() - std::chrono::seconds{10})	{}


        // Convenience constructor for independent graphs
    kernel_graph(int device_number, int stream_id)
        : device_number_(device_number),
          stream_id_(stream_id),
          dependency_number_(INDEPENDENT),
          last_move_(clock_t::now() - std::chrono::seconds{10})	{}


    // returns the next operation/token_ptr that can be dequeued
    token_ptr peek() const {
        if (operations.empty())
            return nullptr;
        return operations.front();
    }

    void add_operation(token_ptr operation) {
        operations.push_back(operation);
    }

    // removes the operation and returns it
    token_ptr getOperation() {
        if (operations.empty())
            return nullptr;

        token_ptr op = operations.front();
        operations.erase(operations.begin());
        return op;
    }

     // Returns true if enough time has passed to allow movement
    bool canMove(std::chrono::seconds min_interval = std::chrono::seconds{2}) const noexcept {
        return (clock_t::now() - last_move_) >= min_interval;
    }

    // Call when the graph is actually moved / rescheduled
    void markMoved() noexcept {
        last_move_ = clock_t::now();
    }

    // Optional: force-disable movement (useful for debugging)
    void disableMove() noexcept {
        last_move_ = clock_t::now() + std::chrono::hours{24};
    }


    bool empty() const {
        return operations.empty();
    }

    int stream_id() const noexcept { return stream_id_; }
    void set_status(int s) noexcept {status = s;}
    int get_status() const noexcept {return status;}


private:
    int device_number_;
    int stream_id_;
    int dependency_number_;
    int status = READY;
    std::vector<token_ptr> operations;
    clock_t::time_point last_move_;
};



struct graph_ref {
    enum class kind_t {
        dependent,
        independent
    };

    kind_t kind;

    // Only valid if kind == dependent
    int dependency = -1;

    // Only valid if kind == independent
    std::size_t index = 0;
};



} // namespace caf::cuda

