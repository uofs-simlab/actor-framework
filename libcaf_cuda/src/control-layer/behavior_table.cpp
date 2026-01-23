#include "caf/cuda/control-layer/behavior_table.hpp"

// IMPORTANT: Include the full definition of the base class
#include "caf/cuda/control-layer/behavior.hpp"  
#include "caf/cuda/control-layer/behavior_token.hpp"  // ← full type here

// If your derived classes have important cleanup, also include them if needed
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include "caf/cuda/control-layer/red_light_behavior.hpp"

namespace caf::cuda {


scheduler_actor_behavior* behavior_table::get(const behavior_token& tok) const {
    auto it = table_.find(tok.name());
    return it != table_.end() ? it->second : nullptr;
}


behavior_table::~behavior_table() {
    table_.clear();  // optional but clean
}

} // namespace caf::cuda
