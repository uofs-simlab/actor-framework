#pragma once
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"

namespace caf::cuda {

class green_light_behavior : public scheduler_actor_behavior {
public:
    void schedule() override {
        // scheduling logic, if needed
    }

    void receive(scheduler_actor_state* state, const token_ptr& tok) override {
        // flush the queue
        while (!state->queue.empty()) {
            auto queued = state->queue.front();
            state->queue.pop();
            // respond to queued token (demo: just print)
        }

        // handle current token immediately
        // (demo: just print)
    }
};

inline green_light_behavior GREEN_BEHAVIOR;

} // namespace caf::cuda

