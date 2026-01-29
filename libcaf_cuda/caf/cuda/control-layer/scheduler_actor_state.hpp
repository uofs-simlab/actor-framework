#pragma once
#include <queue>
#include <vector>
#include "caf/cuda/control-layer/behavior_table.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp" 
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/kernel_graph.hpp"

namespace caf::cuda {
class scheduler_actor_behavior; 

struct scheduler_actor_state {
    caf::actor  self;
    scheduler_actor_behavior* current_behavior = nullptr;
    behavior_table table;
    std::queue<token_ptr> queue; // here for legacy prototype schedulers
    std::vector<kernel_graph> operations; //more modern dependency based data structure 
    int device_number;
    int num_streams; // number of streams that can be used by the scheduler
};
} // namespace caf::cuda
