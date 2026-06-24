#pragma once
/*
 * Meant to be superclass of all tokens send to the scheduler actor
 * unfortunately got cut mostly due to the fact that 
 * it requires too much refactoring then has time to do
 * Will hopefully come back to this class
 */



#include "caf/cuda/control-layer/token.hpp"


//#define INDEPENDENT -1


namespace caf::cuda {

class CAF_CUDA_EXPORT request_token : public token {
public:
  request_token(int dependency = INDEPENDENT)
    : token(dependency) {}

  // Required for CAF message passing
  request_token() = default;


private:

};

using request_token_ptr = caf::intrusive_ptr<request_token>;

} // namespace caf::cuda

