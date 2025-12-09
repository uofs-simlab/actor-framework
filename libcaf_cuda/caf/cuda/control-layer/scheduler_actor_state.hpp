#pragma once
#include <queue>
#include "caf/cuda/control-layer/behavior_table.hpp"
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include "caf/cuda/control-layer/token.hpp"

namespace caf::cuda {

struct scheduler_actor_state {
    scheduler_actor_behavior* current_behavior = nullptr;
    behavior_table table;
    std::queue<token_ptr> queue;
};

} // namespace caf::cuda

