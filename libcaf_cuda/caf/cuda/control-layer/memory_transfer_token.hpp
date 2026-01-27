#pragma once

#include "caf/cuda/control-layer/request_token.hpp"
/*
 * This token represents transfer of memory 
 * from either host or device 
 * all size should be in bytes
 */


// direction defines
#define H2D 1
#define D2H 2

namespace caf::cuda {

class CAF_CUDA_EXPORT memory_transfer_token : public request_token {
public:
  memory_transfer_token(int size,
                        int direction,
                        caf::actor replyActor,
                        int dependency = INDEPENDENT)
    : request_token(dependency),
      size_(size),
      direction_(direction),
      replyActor_(replyActor) {}

  // CAF compliance
  memory_transfer_token() = default;

  int getType() const override { return MEMORY; }

  int getSize() const { return size_; }
  int getDirection() const { return direction_; }
  caf::actor getReplyActor() const { return replyActor_; }

private:
  int size_;
  int direction_;
  caf::actor replyActor_;
};

} // namespace caf::cuda

