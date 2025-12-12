#pragma once

#include "caf/cuda/global_export.hpp" //here to export files
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include <caf/actor.hpp>
#include <string>

namespace caf::cuda {

/// Creates a launch_token (used by users when submitting kernels)
CAF_CUDA_EXPORT token_ptr make_launch_token(program_ptr prog,
                            nd_range range,
                            int memory_usage,
                            std::string id,
                            actor reply_to);

/// Creates a launch_response_token (created internally by the scheduler
/// when it accepts a kernel launch request)
CAF_CUDA_EXPORT token_ptr make_launch_response_token(actor scheduler_or_proxy,
                                     const launch_token& orig);

/// Creates a behavior_token (special — returns its own strong ptr type)
CAF_CUDA_EXPORT behavior_token_ptr make_behavior_token(std::string name);

} // namespace caf::cuda
