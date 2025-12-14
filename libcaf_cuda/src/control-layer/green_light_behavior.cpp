#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include <iostream>

namespace caf::cuda {

    void green_light_behavior::schedule() {
    //TODO IMPLEMENT
    }

    void green_light_behavior::receive(scheduler_actor_state* state, const token_ptr& tok)    {
         
	    std::cout << "Hello from green light processing token\n";

	caf::cuda::launch_token& launch = static_cast<caf::cuda::launch_token&>(*tok);

      // create the response using a reference to the existing object
      caf::cuda::token_ptr r = make_launch_response_token(state->self, launch);

      // send response to the reply actor stored in launch_token
      anon_mail(r).send(launch.getReplyActor());


      std::cout << "GREEN LIGHT DONE PROCESSING TOKEN\n";

	    // flush the queue
    while (!state->queue.empty()) {
        token_ptr queued = state->queue.front();
        state->queue.pop();

 if (queued->getType() == LAUNCH) {
      // safe: we've checked the runtime type
      caf::cuda::launch_token& lt = static_cast<caf::cuda::launch_token&>(*queued);

      // create the response using a reference to the existing object
      caf::cuda::token_ptr response = make_launch_response_token(state->self, lt);

      // send response to the reply actor stored in launch_token
      anon_mail(response).send(lt.getReplyActor());
    }
  }


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

