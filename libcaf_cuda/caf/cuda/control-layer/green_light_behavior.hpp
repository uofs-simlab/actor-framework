#pragma once
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"

namespace caf::cuda {

class green_light_behavior : public scheduler_actor_behavior {
public:
    void schedule() override;
    void receive(scheduler_actor_state* state, const token_ptr& tok) override;
};

// Declare externally defined instance
extern green_light_behavior GREEN_BEHAVIOR;

} // namespace caf::cuda

