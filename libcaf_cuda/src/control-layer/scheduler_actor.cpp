#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include "caf/cuda/control-layer/red_light_behavior.hpp"
#include <string>
#include <iostream>

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
    self->state().current_behavior -> init(&self->state());

    return {
        [=](const token_ptr& tok) {
           

		std::cout << "Received token\n";
		self->state().current_behavior->receive(&self->state(), tok);
        },
        [=](const caf::cuda::behavior_token_ptr& tok) {
            auto* next = self->state().table.get(*tok);
            if (next) {
                
		    self->state().current_behavior -> cleanup(&self->state()); //cleanup current behavior
		    self->state().current_behavior = next;  // swap behavior
		    self->state().current_behavior -> init(&self->state()); //init new current behavior
		
	    }
        },
	[=](std::string word) {
		std::cout << "Received message " << word << "\n";
	},
	 [=](caf::cuda::mem_ptr<int> token) {
            if (!token) {
                std::cout << "Received null mem_ptr\n";
                return;
            }

            if (token->is_scalar()) {
                std::cout << "Received mem_ptr with scalar value: "
                          << *token->host_scalar_ptr() << "\n";
            } else {
                std::cout << "Received mem_ptr with "
                          << token->size() << " elements\n";
                // Optional: print fake data if testing copy_to_host
               // auto host_data = token->copy_to_host();
               // for (auto v : host_data) std::cout << v << " ";
                //std::cout << "\n";
            }
        }

    };
}

} // namespace caf::cuda

