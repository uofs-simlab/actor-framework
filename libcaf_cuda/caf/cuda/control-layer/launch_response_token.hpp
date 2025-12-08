#pragma once
#include "caf/scheduler/token.hpp"
#include "caf/scheduler/launch_token.hpp"
#include <caf/actor.hpp>
#include "caf/cuda/all.hpp"
#include <atomic>

namespace caf::cuda {

class launch_response_token : public token {
public:
    // Construct manually
    launch_response_token(caf::actor receiver,
                          nd_range range,
                          int memory_usage,
                          std::string id)
        : receiver_(std::move(receiver)),
          range_(std::move(range)),
          memory_usage_(memory_usage),
          id_(std::move(id)),
          released_(false) {}

    // Construct from a launch_token
    launch_response_token(caf::actor receiver, const launch_token& token)
        : receiver_(std::move(receiver)),
          range_(token.getRange()),
          memory_usage_(token.getMemoryUsage()),
          id_(token.getId()),
          released_(false) {}

    ~launch_response_token() {
        release();
    }

    int getType() override { return 2; }

    const nd_range& getRange() const { return range_; }
    int getMemoryUsage() const { return memory_usage_; }

    // Return requested number of CUDA blocks
    int getBlocks() const {
        return static_cast<int>(
            range_.getGridDimX() *
            range_.getGridDimY() *
            range_.getGridDimZ()
        );
    }

    const std::string& getId() const { return id_; }

    void release() {
        bool expected = false;
        if (released_.compare_exchange_strong(expected, true)) {
            caf::anon_send(receiver_, id_, getBlocks());
        }
    }

private:
    caf::actor receiver_;
    nd_range range_;
    int memory_usage_;
    std::string id_;
    std::atomic<bool> released_;
};

} // namespace caf::cuda

