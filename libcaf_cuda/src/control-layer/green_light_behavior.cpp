#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"


namespace caf::cuda {

    void green_light_behavior::schedule() {
    //TODO IMPLEMENT
    }

    void green_light_behavior::receive(scheduler_actor_state* state, const token_ptr& tok)    {
        // flush the queue
        while (!state->queue.empty()) {
            auto queued = state->queue.front();
            state->queue.pop();
            // respond to queued token (demo: just print)
        }

        // handle current token immediately
        // (demo: just print)
    }

   void green_light_behavior::init(scheduler_actor_state * state) {
	   std::cout << "GREEN LIGHT\n";
	   behavior_token_ptr red_light = make_behavior_token("red");

	   //send a request to change behavior to green light after 5 seconds
    	   anon_mail(red_light)
		   .delay(std::chrono::seconds(5))
		   .send(state -> self);	
    }



} // namespace caf::cuda

