#pragma once

#include "caf/cuda/control-layer/token.hpp"


#define INDEPENDENT -1


namespace caf::cuda {

class CAF_CUDA_EXPORT request_token : public token {
public:
  request_token(int dependency = INDEPENDENT)
    : dependency_(dependency) {}

  // Required for CAF message passing
  request_token() = default;

  int getDependency() const { return dependency_; }
  bool isIndependent() const { return dependency_ == INDEPENDENT; }

private:
  int dependency_ = INDEPENDENT;
};

using request_token_ptr = caf::intrusive_ptr<request_token>;

} // namespace caf::cuda

