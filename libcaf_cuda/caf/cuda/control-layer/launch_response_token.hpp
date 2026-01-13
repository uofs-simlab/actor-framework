#pragma once
#include "caf/cuda/control-layer/response_token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/global_export.hpp"

#include <caf/actor.hpp>
#include <atomic>
#include <string>
#include "caf/cuda/nd_range.hpp"

namespace caf::cuda {

// -----------------------------------------------------------------------------
// Launch response token returned after a kernel launch request
// -----------------------------------------------------------------------------
class CAF_CUDA_EXPORT launch_response_token : public response_token {
public:
    // Only here to be compliant with CAF's type system – DO NOT USE directly
    launch_response_token() = default;

    // Construct manually
    launch_response_token(caf::actor receiver,
                          nd_range range,
                          int memory_usage,
                          std::string id,
                          int device_num = 0,
                          int stream_id = 0)
        : response_token(std::move(receiver), device_num, stream_id, memory_usage),
          range_(std::move(range)),
          id_(std::move(id)),
          released_(false) {}

    // Construct from a launch_token
    launch_response_token(caf::actor receiver,
                          const launch_token& token,
                          int device_num,
                          int stream_id)
        : response_token(std::move(receiver), device_num, stream_id, token.getMemoryUsage()),
          range_(token.getRange()),
          id_(token.getId()),
          released_(false) {}

    ~launch_response_token() {
        release();
    }

    int getType() const override { return LAUNCH_RESPONSE; }

    const nd_range& getRange() const { return range_; }

    const std::string& getId() const { return id_; }

    // Return requested number of CUDA blocks
    int getBlocks() const {
        return static_cast<int>(
            range_.getGridDimX() *
            range_.getGridDimY() *
            range_.getGridDimZ()
        );
    }

    void release() override {
        bool expected = false;

        // ONLY send if we successfully transition false → true
        if (!released_.compare_exchange_strong(expected, true)) {
            return; // already released → do nothing
        }

        // Real message (commented for testing)
        // caf::anon_mail(id_, memorySize()).urgent().send(receiver_);

        // Test message
        caf::anon_mail("Hello world from launch response").urgent().send(receiver_);
    }

private:
    nd_range range_;
    std::string id_;
    std::atomic<bool> released_;
};

using kernel_launch_token = caf::intrusive_ptr<launch_response_token>;

} // namespace caf::cuda

