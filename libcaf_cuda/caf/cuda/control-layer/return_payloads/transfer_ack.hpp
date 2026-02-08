#pragma once
#include "caf/cuda/control-layer/return_payloads/ack.hpp"

namespace caf::cuda {

class CAF_CUDA_EXPORT transfer_ack final : public ack {
public:
    explicit transfer_ack(int dependency)
      : ack(CAF_CUDA_ACK_TRANSFER),
        dependency_(dependency) {}

    int dependency() const { return dependency_; }

private:
    int dependency_;
};

} //namespace caf::cuda


