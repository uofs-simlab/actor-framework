#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include <iostream>

namespace caf::cuda {

green_light_behavior::green_light_behavior(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {}

void green_light_behavior::on_enter() {
    //std::cout << "GREEN LIGHT\n";

    behavior_token_ptr red_light = make_behavior_token("red");
    //send a request to change behavior to red light after 5 seconds
    /*
    anon_mail(red_light)
            .delay(std::chrono::seconds(5))
            .send(state_.self);
    */
}

void green_light_behavior::schedule() {
    //TODO IMPLEMENT
}

void green_light_behavior::receive(const token_ptr& tok) {
   

//	std::cout << "Green light receive\n";

     if (tok->getType() == LAUNCH) {
        //use 0 as stream id for now, eventually will have to figure out
        //stream load balancing
        process_launch_token(tok, rand()% state_.num_streams);
    }
    else if (tok->getType() == MEMORY) {
        //use 0 as stream id for now, eventually will have to figure out
        //stream load balancing
        process_memory_transfer_token(tok, 0);
    }
    //this may cause an issue if a message is never received then
    //we may never end up dequeueing certain requests
    //may lead to a deadlock scenario?
   /*
    while (!state_.queue.empty()) {
        token_ptr queued = state_.queue.front();
        state_.queue.pop();
        if (queued->getType() == LAUNCH) {
            // safe: we've checked the runtime type
            process_launch_token(queued, 0);
        }
        else if (queued->getType() == MEMORY) {
            process_memory_transfer_token(queued, 0);
        }
    }
    */
}


} // namespace caf::cuda
