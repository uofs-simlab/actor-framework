#pragma once
#include <caf/all.hpp>
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/behavior_table.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include <queue>
#include "caf/cuda/global_export.hpp"

/*
 * The scheduler actor 
 * meant to make scheduling decisions using s/r/r ipc
 */

namespace caf::cuda {
caf::behavior CAF_CUDA_EXPORT scheduler_actor(caf::stateful_actor<scheduler_actor_state> * self,int device_number);
}//namespace caf::cuda
