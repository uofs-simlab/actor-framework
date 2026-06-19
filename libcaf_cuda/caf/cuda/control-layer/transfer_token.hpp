#pragma once
#include "caf/cuda/control-layer/response_token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
// #include "caf/cuda/control-layer/return_payloads/transfer_ack.hpp"
#include "caf/cuda/global_export.hpp"

#include <caf/actor.hpp>
#include <atomic>
#include <string>
#include "caf/cuda/nd_range.hpp"

namespace caf::cuda {

// -----------------------------------------------------------------------------
// Transfer response token returned after a kernel transfer request
// -----------------------------------------------------------------------------
class CAF_CUDA_EXPORT transfer_token : public response_token {
public:
    // Default constructor – only for CAF compliance
    transfer_token() = default;

    // Construct manually (full control over dependency number)
    transfer_token(caf::actor receiver,
                   nd_range range,
                   int memory_usage,
                   std::string id,
                   int device_num = 0,
                   int stream_id = 0,
                   int dependency_number = 0)
        : response_token(std::move(receiver), device_num, stream_id, memory_usage),
          range_(std::move(range)),
          id_(std::move(id)),
          released_(false),
          dependency_number_(dependency_number) {}

    // Construct from a launch_token
    transfer_token(caf::actor receiver,
                   const launch_token& token,
                   int device_num,
                   int stream_id)
        : response_token(std::move(receiver),
                         device_num,
                         stream_id,
                         token.getMemoryUsage()),
          range_(token.getRange()),
          id_(token.getId()),
          released_(false),
          dependency_number_(token.getDependency()) {}

    ~transfer_token() override {
        release();
    }

    int getType() const override { return TRANSFER; }

    const nd_range& getRange() const { return range_; }
    const std::string& getId() const { return id_; }
    const std::string& name() const override { return id_; }

    // -------------------------------------------------------------------------
    // Release and send transfer_ack as base ack (explicit upcast)
    // -------------------------------------------------------------------------
    void release() override {
        bool expected = false;
        if (!released_.compare_exchange_strong(expected, true))
            return;

        try {
        //     // Create concrete transfer_ack
        //     transfer_ack ack_obj{dependency_number_};

        //     // Upcast explicitly to ack reference before sending
        //     [[maybe_unused]] const ack& base_ack = ack_obj;
	    // caf::anon_mail(std::move(ack_obj)).urgent().send(receiver_);


        } catch (...) {
            // destructor-safe
        }
    }

private:
    nd_range range_;
    std::string id_;
    std::atomic<bool> released_{false};
    int dependency_number_{0};
};

} // namespace caf::cuda
