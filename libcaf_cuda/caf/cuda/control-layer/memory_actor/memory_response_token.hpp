#pragma once
#include <caf/actor.hpp>
#include "caf/cuda/control-layer/return_payloads/ack.hpp"
#include "caf/cuda/global_export.hpp"
#include <utility>

namespace caf::cuda {

/// RAII-style memory tracker that returns memory to the memory actor
/// when it goes out of scope.
class CAF_CUDA_EXPORT memory_response_token {
public:
    memory_response_token(std::size_t memory_in_bytes,
                          int device_number,
                          caf::actor memory_actor_handle) 
        : memory_bytes_(memory_in_bytes),
          device_number_(device_number),
          memory_actor_(std::move(memory_actor_handle)) {}

    // Move constructor (allow moving, but disable copying)
    memory_response_token(memory_response_token&& other) noexcept
        : memory_bytes_(other.memory_bytes_),
          device_number_(other.device_number_),
          memory_actor_(std::move(other.memory_actor_)) 
    {
        other.memory_bytes_ = 0;  // prevent double return
    }

    memory_response_token& operator=(memory_response_token&& other) noexcept {
        if (this != &other) {
            memory_bytes_ = other.memory_bytes_;
            device_number_ = other.device_number_;
            memory_actor_ = std::move(other.memory_actor_);
            other.memory_bytes_ = 0;  // prevent double return
        }
        return *this;
    }

    // Delete copy constructor/assignment to avoid double returns
    memory_response_token(const memory_response_token&) = delete;
    memory_response_token& operator=(const memory_response_token&) = delete;

    ~memory_response_token() {
        if (memory_bytes_ > 0 && memory_actor_) {
            // RAII return: tell memory actor we freed memory
            anon_mail(ack(CAF_CUDA_ACK_MEMORY))
                .arg(device_number_)
                .arg(memory_bytes_)
                .send(memory_actor_);
        }
    }

private:
    std::size_t memory_bytes_;
    int device_number_;
    caf::actor memory_actor_;
};

} // namespace caf::cuda
