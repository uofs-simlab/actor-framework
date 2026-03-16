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

//  std::cout << "Green light receive\n";

    if (tok->getType() == LAUNCH) {

        const auto& launch = static_cast<const launch_token&>(*tok);

        int stream_id = rand() % state_.num_streams;

        // Manually construct launch_response_token with anon_mail disabled
        response_token_ptr response(
            new launch_response_token(
                state_.self,
                launch,
                state_.device_number,
                stream_id,
                0,      // reclaim_value
                0,      // reclaim_runtime
                false   // send_mail disabled
            )
        );

        anon_mail(response).send(launch.getReplyActor());

        //(void)response; // suppress unused warning
    }
    else if (tok->getType() == MEMORY) {
        process_memory_transfer_token(tok, 0);
    }
}


} // namespace caf::cuda
