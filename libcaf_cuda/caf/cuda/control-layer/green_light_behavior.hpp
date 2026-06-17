#pragma once
#include "caf/cuda/control-layer/behavior.hpp"

namespace caf::cuda {

class green_light_behavior : public scheduler_actor_behavior {
public:
    explicit green_light_behavior(scheduler_actor_state& state);
    void on_enter() override;
    void schedule() override;
    void receive([[maybe_unused]] const token_ptr& tok) override;
    std::string name() const override {return "green\n";}
};

} // namespace caf::cuda
