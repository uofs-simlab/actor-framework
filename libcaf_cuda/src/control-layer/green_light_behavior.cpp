#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include <iostream>

namespace caf::cuda {

    void green_light_behavior::schedule() {
    //TODO IMPLEMENT
    }

    void green_light_behavior::receive(scheduler_actor_state* state, const token_ptr& tok)    {
         

	if (tok -> getType() == LAUNCH) {
	caf::cuda::launch_token& launch = static_cast<caf::cuda::launch_token&>(*tok);

      // create the response using a reference to the existing object
      caf::cuda::token_ptr r = make_launch_response_token(state->self, launch);

      // send response to the reply actor stored in launch_token
      anon_mail(r).send(launch.getReplyActor());
	
     }

	else if (tok -> getType() == MEMORY) {
	

	
	}

	//this may cause an issue if a message is never received then 
	//we may never end up dequeueing certain requests 
	//may lead to a deadlock scenario?
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

//--------------------------------------------------
// process_launch_token
//--------------------------------------------------
void green_light_behavior::process_launch_token(const token_ptr& tok,caf::actor self) {
    caf::cuda::launch_token& launch =
        static_cast<caf::cuda::launch_token&>(*tok);

    // Create response token using the existing launch token
    caf::cuda::token_ptr response =
        make_launch_response_token(self, launch);

    // Send response to the actor that requested the launch
    anon_mail(response).send(launch.getReplyActor());
}

//--------------------------------------------------
// process_memory_transfer_token
//--------------------------------------------------
void green_light_behavior::process_memory_transfer_token(const token_ptr& tok,
                                                         caf::actor self) {
    caf::cuda::memory_transfer_token& mem =
        static_cast<caf::cuda::memory_transfer_token&>(*tok);

    // Create memory response token
    caf::cuda::token_ptr response =
        make_memory_response_token(self, mem);

    // Send response to requesting actor
    anon_mail(response).send(mem.getReplyActor());
}



} // namespace caf::cuda

