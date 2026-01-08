#pragma once
#include "caf/cuda/control-layer/behavior.hpp"

namespace caf::cuda {

class green_light_behavior : public scheduler_actor_behavior {
public:
    void schedule() override;

    void receive(scheduler_actor_state* state, const token_ptr& tok) override; 
    void init(scheduler_actor_state* state); 
    void process_launch_token(const token_ptr& tok);
    void process_memory_transfer_token(const token_ptr& tok);
};


} // namespace caf::cuda

