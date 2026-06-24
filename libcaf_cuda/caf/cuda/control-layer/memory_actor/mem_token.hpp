#pragma once
#include <caf/actor.hpp>
// #include "caf/cuda/control-layer/return_payloads/ack.hpp"
#include "caf/cuda/global_export.hpp"
#include <utility>

namespace caf::cuda {

/// RAII-style memory tracker that returns memory to the memory actor
/// when it goes out of scope.
class CAF_CUDA_EXPORT mem_token {
public:
    mem_token() = default;

    mem_token(std::size_t memory_in_bytes,
              int device_number,
              caf::actor memory_actor_handle)
        : memory_bytes_(memory_in_bytes),
          device_number_(device_number),
          memory_actor_(std::move(memory_actor_handle)) {}

    // Copy constructor: transfer ownership
    mem_token(const mem_token& other)
        : memory_bytes_(other.memory_bytes_),
          device_number_(other.device_number_),
          memory_actor_(other.memory_actor_) 
    {
        const_cast<mem_token&>(other).memory_bytes_ = 0;
    }

    // Copy assignment: transfer ownership
    mem_token& operator=(const mem_token& other) {
        if (this != &other) {
            // First, return any existing memory
            release();
            memory_bytes_ = other.memory_bytes_;
            device_number_ = other.device_number_;
            memory_actor_ = other.memory_actor_;
            const_cast<mem_token&>(other).memory_bytes_ = 0;
        }
        return *this;
    }

    // Move constructor
    mem_token(mem_token&& other) noexcept
        : memory_bytes_(other.memory_bytes_),
          device_number_(other.device_number_),
          memory_actor_(std::move(other.memory_actor_))
    {
        other.memory_bytes_ = 0;
    }

    // Move assignment
    mem_token& operator=(mem_token&& other) noexcept {
        if (this != &other) {
            release();
            memory_bytes_ = other.memory_bytes_;
            device_number_ = other.device_number_;
            memory_actor_ = std::move(other.memory_actor_);
            other.memory_bytes_ = 0;
        }
        return *this;
    }

    ~mem_token() { release(); }

private:
    void release() {
        if (memory_bytes_ > 0 && memory_actor_) {
            anon_mail(device_number_, memory_bytes_).send(memory_actor_);
            memory_bytes_ = 0;
        }
    }

    std::size_t memory_bytes_ = 0;
    int device_number_ = -1;
    caf::actor memory_actor_;
};

} // namespace caf::cuda
