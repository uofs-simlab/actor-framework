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
    std::unique_ptr<behavior_table> table;
    //behavior_table table;
    std::queue<token_ptr> queue; // here for legacy prototype schedulers
    std::vector<kernel_graph> operations; //more modern dependency based data structure 
    int device_number;
    int in_flight = 0; // current number of tasks active on the GPU
    int num_streams = 500;
    int stream_depth = 1; // maximum number of in-flight tasks per stream
    std::vector<caf::actor> schedulers; //the other scheduler actors in the system, an actor for a 
			     //specific GPU can be accessed via there corrosponding 
			     //device number in the std::vector
   bool multiple_gpus = false; //flag that will indicate to the actor whether or not there is
			       //multiple GPUs 

};
} // namespace caf::cuda
