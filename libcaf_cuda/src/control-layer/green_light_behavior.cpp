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

} // namespace caf::cuda

