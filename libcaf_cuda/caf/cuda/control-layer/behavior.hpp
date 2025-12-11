#pragma once 
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/scheduler_actor_state.hpp"


//this class is meant to provide an interface so that 
//the actor can change behavior at runtime
//normally I would say use become 
//however that is too much boiler plate 
//so instead we can just build a behavior
//abstract class
//and create a dispatch table

namespace caf::cuda {

struct scheduler_actor_state;
class launch_token;

class scheduler_actor_behavior {
public:
    virtual ~scheduler_actor_behavior() = default;

    virtual void schedule() = 0;
    virtual void receive(scheduler_actor_state* state, const token_ptr& tok) = 0;
    
    //define what to do when transitioning into the state
    virtual void init(scheduler_actor_state * state) {
    	//If the behavior decides not to overide do nothing
    }

    //define what to do when transitioning out of the state
    virtual void cleanup(scheduler_actor_state * state) { 
    	//If the behavior decides not to overide do nothing
    }
};

} // namespace caf::cuda

