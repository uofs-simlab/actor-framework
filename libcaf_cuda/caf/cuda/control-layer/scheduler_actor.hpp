#pragma once
#include <caf/all.hpp>
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/behavior_table.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include <queue>

/*
 * The scheduler actor 
 * meant to make scheduling decisions using s/r/r ipc
 */

namespace caf::cuda {
caf::behavior scheduler_actor(caf::stateful_actor<scheduler_actor_state> * self);
}//namespace caf::cuda
