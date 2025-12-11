#pragma once
#include "caf/cuda/control-layer/behavior.hpp"

namespace caf::cuda {

class green_light_behavior : public scheduler_actor_behavior {
public:
    void schedule() override;

    void receive(scheduler_actor_state* state, const token_ptr& tok) override; 
    void init(scheduler_actor_state* state); 
};


} // namespace caf::cuda

