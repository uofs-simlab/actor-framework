#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include "caf/cuda/control-layer/red_light_behavior.hpp"
#include <string>

/*
 * This class is meant to handle actor GPU scheduling via s/r/r IPC 
 * it has nothing to do with the scheduler class, that is kernel laye
 */ 

namespace caf::cuda {

caf::behavior scheduler_actor(caf::stateful_actor<scheduler_actor_state>* self) {

    //add its self reference
    self -> state().self = self;

    // populate the table
    static red_light_behavior red_behavior;
    static green_light_behavior green_behavior;

    self->state().table.add("green", &green_behavior);
    self->state().table.add("red", &red_behavior);

    // default behavior
    self->state().current_behavior = self->state().table.get(behavior_token("green"));

    return {
        [=](const token_ptr& tok) {
            self->state().current_behavior->receive(&self->state(), tok);
        },
        [=](const behavior_token& tok) {
            auto* next = self->state().table.get(tok);
            if (next) {
                
		    self->state().current_behavior -> cleanup(&self->state()); //cleanup current behavior
		    self->state().current_behavior = next;  // swap behavior
		    self->state().current_behavior -> init(&self->state()); //init new current behavior
		
	    }
        },
	[=](std::string word) {
		std::cout << "Received message " << word << "\n";
	}
    };
}

} // namespace caf::cuda

