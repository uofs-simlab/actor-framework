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
    // Default constructor – only for CAF compliance
    launch_response_token() = default;

    // Construct manually (full control over reclaim fields)
    launch_response_token(caf::actor receiver,
                          nd_range range,
                          int memory_usage,
                          std::string id,
                          int device_num = 0,
                          int stream_id = 0,
                          int reclaim_value = 0,
                          int reclaim_memory_returned = 0,
                          int reclaim_runtime = 0,
                          int reclaim_dependency = 0,
                          bool send_mail = true)
        : response_token(std::move(receiver), device_num, stream_id, memory_usage),
          range_(std::move(range)),
          id_(std::move(id)),
          released_(false),
          reclaim_value_(reclaim_value),
          reclaim_memory_returned_(reclaim_memory_returned),
          reclaim_runtime_(reclaim_runtime),
          reclaim_dependency_(reclaim_dependency),
          send_mail_(send_mail) {}

    // Construct from a launch_token
    // Memory returned and dependency are copied from launch_token
    launch_response_token(caf::actor receiver,
                          const launch_token& token,
                          int device_num,
                          int stream_id,
                          int reclaim_value = 0,
                          int reclaim_runtime = 0,
                          bool send_mail = true)
        : response_token(std::move(receiver), device_num, stream_id, token.getMemoryUsage()),
          range_(token.getRange()),
          id_(token.getId()),
          released_(false),
          reclaim_value_(reclaim_value),
          reclaim_memory_returned_(token.getMemoryUsage()),
          reclaim_runtime_(reclaim_runtime),
          reclaim_dependency_(token.getDependency()),
          send_mail_(send_mail) {}

    ~launch_response_token() {
        release();
    }

    int getType() const override { return LAUNCH_RESPONSE; }

    const nd_range& getRange() const { return range_; }

    const std::string& getId() const { return id_; }

    const std::string& name() const override { return id_; }

    // Return requested number of CUDA blocks
    int getBlocks() const {
        return static_cast<int>(
            range_.getGridDimX() *
            range_.getGridDimY() *
            range_.getGridDimZ()
        );
    }

    // Release and send reclaim information exactly once
    void release() override {
        bool expected = false;

        if (!released_.compare_exchange_strong(expected, true)) {
            return; // already released
        }

        if (!send_mail_) {
            return; // mail sending disabled
        }

        try {
            caf::anon_mail(
                reclaim_value_,
                reclaim_memory_returned_,
                reclaim_runtime_,
                reclaim_dependency_,
                stream_id_
            ).urgent().send(receiver_);
        } catch (...) {
            // swallow exceptions — destructor safe
        }
    }

    // Accessors for reclaim fields
    int reclaim_value() const { return reclaim_value_; }
    int reclaim_memory_returned() const { return reclaim_memory_returned_; }
    int reclaim_runtime() const { return reclaim_runtime_; }
    int reclaim_dependency() const { return reclaim_dependency_; }

private:
    nd_range range_;
    std::string id_;
    std::atomic<bool> released_{false};

    // Reclaim fields
    int reclaim_value_ = 0;
    int reclaim_memory_returned_ = 0;
    int reclaim_runtime_ = 0;
    int reclaim_dependency_ = 0;

    // Toggle for sending anon_mail
    bool send_mail_ = true;
};

using kernel_launch_token = caf::intrusive_ptr<launch_response_token>;

} // namespace caf::cuda
