#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"


namespace caf::cuda {

    void green_light_behavior::schedule() {
    //TODO IMPLEMENT
    }

    void green_light_behavior::receive(scheduler_actor_state* state, const token_ptr& tok)    {
         // flush the queue
    while (!state->queue.empty()) {
        token_ptr queued = state->queue.front();
        state->queue.pop();

        if (queued->getType() == LAUNCH) {
            // manually downcast raw pointer
            caf::intrusive_ptr<launch_token> ltok(static_cast<launch_token*>(queued.get()));

            // dereference to pass reference to factory function
            token_ptr response = make_launch_response_token(state->self, *ltok);
            anon_mail(response).send(ltok->getReplyActor());
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

