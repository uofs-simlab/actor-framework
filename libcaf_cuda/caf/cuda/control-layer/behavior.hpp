#pragma once
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/memory_transfer_token.hpp"
#include "caf/cuda/control-layer/scheduler_actor_state.hpp"
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/token_factory.hpp"




//this class is meant to provide an interface so that 
//the actor can change behavior at runtime
//normally I would say use become 
//however that is too much boiler plate 
//so instead we can just build a behavior
//abstract class
//and create a dispatch table

namespace caf::cuda {

class scheduler_actor_behavior {
public:
    explicit scheduler_actor_behavior(scheduler_actor_state& state)
        : state_(state) {}
    virtual ~scheduler_actor_behavior() = default;

    virtual void on_enter() {}
    virtual void on_exit() {}

    virtual void schedule() = 0;
    virtual void receive(const token_ptr& tok) = 0;

protected:
    scheduler_actor_state& state_;

    // Default implementation (immediate response) – takes token_ptr and casts internally
    virtual void process_launch_token(const token_ptr& tok, int stream_id);
    virtual void process_memory_transfer_token(const token_ptr& tok, int stream_id);
};

} // namespace caf::cuda

