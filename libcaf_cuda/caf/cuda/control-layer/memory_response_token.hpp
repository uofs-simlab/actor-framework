#pragma once
#include "caf/cuda/control-layer/response_token.hpp"
#include "caf/cuda/control-layer/memory_transfer_token.hpp"
#include "caf/cuda/global_export.hpp"

#include <caf/actor.hpp>
#include <atomic>

namespace caf::cuda {

// -----------------------------------------------------------------------------
// Memory response issued by the scheduler / actor authorizing memory transfer
// -----------------------------------------------------------------------------
class CAF_CUDA_EXPORT memory_response_token : public response_token {
public:
    // Required by CAF – do not use directly
    memory_response_token() = default;

    // Construct from memory_transfer_token
    memory_response_token(caf::actor receiver,
                          const memory_transfer_token& token,
                          int device_num,
                          int stream_id)
        : response_token(std::move(receiver), device_num, stream_id, token.getSize()),
          direction_(token.getDirection()),
          released_(false) {}

    ~memory_response_token() {
        release();
    }

    int getType() const override {
        return MEMORY_RESPONSE;
    }

    int getDirection() const {
        return direction_;
    }

    void release() override {
        bool expected = false;

        // ONLY send if we successfully transition false → true
        if (!released_.compare_exchange_strong(expected, true)) {
            return; // already released → do nothing
        }

        // Real message (commented for testing)
        // caf::anon_mail(memorySize(), direction_).urgent().send(receiver_);

        // Test message
        caf::anon_mail("Hello world from memory response")
            .urgent()
            .send(receiver_);
    }

private:
    int direction_;
    std::atomic<bool> released_;
};

using memory_token = caf::intrusive_ptr<memory_response_token>;

} // namespace caf::cuda

