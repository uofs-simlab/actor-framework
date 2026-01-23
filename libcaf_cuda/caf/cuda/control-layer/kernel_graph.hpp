#pragma once
#include <vector>
#include "caf/cuda/control-layer/token.hpp"

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
    // for caf's messaging system do not use
    kernel_graph() = default;

    kernel_graph(int device_number,
                 int stream_id,
                 int dependency_number)
        : device_number_(device_number),
          stream_id_(stream_id),
          dependency_number_(dependency_number) {}

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

    bool empty() const {
        return operations.empty();
    }

private:
    int device_number_;
    int stream_id_;
    int dependency_number_;
    std::vector<token_ptr> operations;
};

} // namespace caf::cuda

