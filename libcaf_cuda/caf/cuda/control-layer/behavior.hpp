#pragma once
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/memory_transfer_token.hpp"
#include "caf/cuda/control-layer/transfer_token.hpp"
#include "caf/cuda/control-layer/scheduler_actor_state.hpp"
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/return_payloads/all_return_payloads.hpp"

#include <string>



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

    virtual void reclaim([[maybe_unused]] int value, [[maybe_unused]] int memory_returned, [[maybe_unused]] int runtime, [[maybe_unused]] int dependency) {
	    //default implementation is to do nothing, this should be overidden
	    //by children classes 
    }



    //this is here to ensure that payloads on return can conform to an interface
    //rather than changing the interface to accomidate every scheduling need
    virtual void reclaim([[maybe_unused]] ack& payload) {
	    //default implementation is to do nothing, this should be overidden
	    //by children classes 
    }


    virtual std::string name() const {return "No name\n";}

    //this method is meant to be a handler for when
    //another scheduler actor queries for more work
    virtual void handle_load_balance_request([[maybe_unused]] int device_number) {
    
	    //default action is to do nothing and not particpate in load balancing
	    //whether of not a scheduler wants to participate in load balancing 
	    //and what actions it should take is a policy decision
    }


    //method is meant to handle work being sent over from another scheduler actor
   virtual void receive_work([[maybe_unused]] std::vector<kernel_graph> work_graphs) {
     //ideally this should not default to do nothing 
    //however I do not have the time implement this on every existing behavior 
    //as of right now
    //so be warned if you do not implement an override and request work to do be done
    //this will end in a deadlock  
    
    
    }




protected:
    scheduler_actor_state& state_;

    // Default implementation (immediate response) – takes token_ptr and casts internally
    virtual void process_launch_token(const token_ptr& tok, int stream_id);
    virtual void process_memory_transfer_token(const token_ptr& tok, int stream_id);
    virtual void dispatch_transfer_token(const token_ptr& tok, int stream_id);
};

} // namespace caf::cuda
