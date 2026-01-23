#pragma once
#include "caf/cuda/control-layer/behavior.hpp"

namespace caf::cuda {

class red_light_behavior : public scheduler_actor_behavior {
public:
    explicit red_light_behavior(scheduler_actor_state& state);
    void on_enter() override;
    void schedule() override;
    void receive(const token_ptr& tok) override;
    ~red_light_behavior() noexcept override;

};

} // namespace caf::cuda
