#include "caf/cuda/control-layer/red_light_behavior.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include <iostream>

namespace caf::cuda {

// Define the member functions for red_light_behavior
void red_light_behavior::schedule() {
    // scheduling logic if needed
}

void red_light_behavior::receive(scheduler_actor_state* state,
                                 const token_ptr& tok) {
    state->queue.push(tok);
}

// Define the global instance
red_light_behavior RED_BEHAVIOR;

// -------------------------------------------

// Define the member functions for green_light_behavior
void green_light_behavior::schedule() {
    // scheduling logic if needed
}

void green_light_behavior::receive(scheduler_actor_state* state,
                                   const token_ptr& tok) {
    // flush the queue
    while (!state->queue.empty()) {
        auto queued = state->queue.front();
        state->queue.pop();
        // respond to queued token (demo: just print)
        std::cout << "Processing queued token\n";
    }

    // handle current token immediately
    std::cout << "Processing current token\n";
}

// Define the global instance
green_light_behavior GREEN_BEHAVIOR;

} // namespace caf::cuda

