#pragma once
#include "caf/cuda/control-layer/behavior.hpp"

namespace caf::cuda {

class red_light_behavior : public scheduler_actor_behavior {
public:
    void schedule() override {
        // scheduling logic, if needed
    }

    void receive(scheduler_actor_state* state, const token_ptr& tok) override;

    ~red_light_behavior() noexcept override;
};


} // namespace caf::cuda

