#pragma once
#include "caf/cuda/control-layer/red_light_behavior.hpp"

namespace caf::cuda {

    void red_light_behavior::schedule() override {
        // scheduling logic, if needed
    }

    void red_light_behavior::receive(scheduler_actor_state* state, const token_ptr& tok) override {
        state->queue.push(tok); // enqueue everything
    }

    red_light_behavior::~red_light_behavior() noexcept = default;

} // namespace caf::cuda
