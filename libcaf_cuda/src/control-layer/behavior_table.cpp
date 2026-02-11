#include "caf/cuda/control-layer/behavior_table.hpp"

#include "caf/cuda/control-layer/behavior.hpp"  
#include "caf/cuda/control-layer/behavior_token.hpp"  

// If your derived classes have important cleanup, also include them if needed
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include "caf/cuda/control-layer/red_light_behavior.hpp"
#include "caf/cuda/control-layer/core_usage_behavior.hpp"
#include "caf/cuda/control-layer/single_usage_behavior.hpp"
#include "caf/cuda/control-layer/multilevel_usage_behavior.hpp"
#include "caf/cuda/control-layer/pressure_scheduler.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"

namespace caf::cuda {

 // Constructor that creates all default behaviors for a given actor state
    behavior_table::behavior_table(scheduler_actor_state& state) {
        // dynamically allocate behaviors and add to table
        add("red",          new red_light_behavior(state));
        add("green",        new green_light_behavior(state));
        add("core_usage",   new core_usage_behavior(state));
        add("single_usage", new single_usage_behavior(state));
        add("multilevel",   new multilevel_usage_behavior(state));
        add("pressure",     new pressure_scheduler(state));
    }


    behavior_table::~behavior_table() {
        for (auto& [name, beh] : table_)
            delete beh;  // clean up all behaviors on destruction
    }


scheduler_actor_behavior* behavior_table::get(const behavior_token& tok) const {
    auto it = table_.find(tok.name());
    return it != table_.end() ? it->second : nullptr;
}


} // namespace caf::cuda
