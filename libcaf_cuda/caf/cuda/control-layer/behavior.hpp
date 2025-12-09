#pragma once 
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"



//this class is meant to provide an interface so that 
//the actor can change behavior at runtime
//normally I would say use become 
//however that is too much boiler plate 
//so instead we can just build a behavior
//abstract class
//and create a dispatch table

namespace caf::cuda {

class launch_token;

class scheduler_actor_behavior {
public:
    virtual ~scheduler_actor_behavior() = default;

    virtual void schedule() = 0;
    virtual void receive(class scheduler_actor_state* state, const launch_token& tok) = 0;
};

} // namespace caf::cuda

