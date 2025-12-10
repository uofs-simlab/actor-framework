#pragma once
#include "caf/cuda/control-layer/behavior.hpp"

namespace caf::cuda {

class red_light_behavior : public scheduler_actor_behavior {
public:
    void schedule() override;
    void receive(scheduler_actor_state* state, const token_ptr& tok) override;
};

// Declare externally defined instance
extern red_light_behavior RED_BEHAVIOR;

} // namespace caf::cuda

