#pragma once
#include <queue>
#include "caf/cuda/control-layer/behavior_table.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"  // Still needed for behavior_token
#include "caf/cuda/control-layer/token.hpp"

namespace caf::cuda {
class scheduler_actor_behavior; 

struct scheduler_actor_state {
    caf::actor  self_handle;
    scheduler_actor_behavior* current_behavior = nullptr;
    behavior_table table;
    std::queue<token_ptr> queue;
};
} // namespace caf::cuda
