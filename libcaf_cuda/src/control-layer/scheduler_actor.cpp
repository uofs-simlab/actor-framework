#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include "caf/cuda/control-layer/red_light_behavior.hpp"
#include "caf/cuda/control-layer/core_usage_behavior.hpp"
#include "caf/cuda/control-layer/single_usage_behavior.hpp"
#include "caf/cuda/control-layer/multilevel_usage_behavior.hpp"
#include "caf/cuda/control-layer/pressure_scheduler.hpp"
#include "caf/cuda/control-layer/kernel_graph.hpp"
#include <string>
#include <iostream>

/*
 * This class is meant to handle actor GPU scheduling via s/r/r IPC
 * it has nothing to do with the scheduler class, that is kernel layer
 */
namespace caf::cuda {

caf::behavior scheduler_actor(caf::stateful_actor<scheduler_actor_state>* self, int device_number,bool multi_gpu) {
    auto& state = self->state();

    // add self reference
    state.self = self;

    // set device number
    state.device_number = device_number;

    //check if multiple gpus
    state.multiple_gpus = multi_gpu;

    //static declarations may cause issues when expanding to
    //multiple GPUs
    static red_light_behavior red_behavior(state);
    static green_light_behavior green_behavior(state);
    static core_usage_behavior core_behavior(state);
    static single_usage_behavior single_behavior(state);
    static multilevel_usage_behavior multi_behavior(state);
    static pressure_scheduler pressure(state);

    // populate the behavior table
    state.table.add("red", &red_behavior);
    state.table.add("green",   &green_behavior);
    state.table.add("core_usage",   &core_behavior);
    state.table.add("multilevel",   &multi_behavior);
    state.table.add("pressure",   &pressure);

    // default behavior
    state.current_behavior = state.table.get(behavior_token("green"));
    state.current_behavior->on_enter();



    return {
        [&](const token_ptr& tok) {
            // std::cout << "Received token\n";
            state.current_behavior->receive(tok);
        },
        
   [&state](const caf::cuda::behavior_token_ptr& tok) -> bool {
    auto* next = state.table.get(*tok);
    if (next) {
        if (next != state.current_behavior) {
            state.current_behavior->on_exit();   // cleanup current behavior
            state.current_behavior = next;       // swap behavior
            state.current_behavior->on_enter();  // init new behavior
            std::cout << "[INFO] Behavior changed to: " << state.current_behavior->name() << "\n";
	    return true; // behavior changed
        } else {
            std::cout << "[INFO] Behavior already active: " << state.current_behavior->name() << "\n";
            return false; // behavior was already current
        }
    } else {
        std::cout << "[WARN] No behavior found for token: " << tok->name() << "\n";
        return false; // no change
    }
	}
	,
        [&](std::vector<token_ptr> tokens) {
            for (size_t i = 0; i < tokens.size(); ++i) {
                state.current_behavior->receive(tokens[i]);
            }
        },

	//can send the scheduler a message if you want 
	//it is more than happy to print it out for you 
        [=](std::string word) {
             std::cout << "Received message " << word << "\n";
        },
	
	//message handler for reclaim
	[&](int value, int memory,int runtime,int dependency) {	
		state.current_behavior->reclaim(value,memory,runtime,dependency);
	},


	//message handler for reclaim
	[&](ack payload) {	
		state.current_behavior->reclaim(payload);
	},




	//handler sent to set the scheduler actors 
	//do not send a message more than once
	//or else undefined behavior
	[&](std::vector<caf::actor> s) {
		state.schedulers = s;
	},



	//message handler for a request for work from another scheduler actor
	[&](int device_number) {
		state.current_behavior -> handle_load_balance_request(device_number);
	},

	//message handler for work being transfered over from another scheduler actor
	[&](std::vector<kernel_graph> work_graphs) {
		state.current_behavior -> receive_work(work_graphs);
	},
    };
}

} // namespace caf::cuda
