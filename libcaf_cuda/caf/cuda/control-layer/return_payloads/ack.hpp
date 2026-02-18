#pragma once
#include "caf/cuda/global_export.hpp"

namespace caf::cuda {

// -----------------------------------------------------------------------------
// ACK type codes (stable ABI, no enum churn)
// -----------------------------------------------------------------------------
#define CAF_CUDA_ACK_TRANSFER  1
#define CAF_CUDA_ACK_LAUNCH    2
#define CAF_CUDA_ACK_MEMORY    3
#define TIMER 4


// -----------------------------------------------------------------------------
// Base ACK payload
// -----------------------------------------------------------------------------
class CAF_CUDA_EXPORT ack {
public:
   
   //for caf messaging system do not use	
    ack() = default;
    explicit ack(int type) : type_(type) {}
    virtual ~ack() = default;

    int getType() const { return type_; }

private:
    int type_;
};

} // namespace caf::cuda

