#pragma once
#include "caf/cuda/control-layer/behavior.hpp"

namespace caf::cuda {

class core_usage_behavior : public scheduler_actor_behavior {
public:
    explicit core_usage_behavior(scheduler_actor_state& state);
    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;

};

} // namespace caf::cuda

