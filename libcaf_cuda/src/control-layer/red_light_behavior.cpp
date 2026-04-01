#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/red_light_behavior.hpp"
#include "caf/cuda/control-layer/token_factory.hpp"
#include <iostream>

namespace caf::cuda {

red_light_behavior::red_light_behavior(scheduler_actor_state& state)
    : scheduler_actor_behavior(state) {}

void red_light_behavior::schedule() {}

void red_light_behavior::receive(const token_ptr& tok) {
    state_.queue.push(tok); // enqueue everything
}

red_light_behavior::~red_light_behavior() noexcept = default;

void red_light_behavior::on_enter() {
    std::cout << "RED LIGHT\n";
    behavior_token_ptr green_light = make_behavior_token("green");
    //send a request to change behavior to green light after 5 seconds
    anon_mail(green_light)
            .delay(std::chrono::seconds(5))
            .send(state_.self);
}

} // namespace caf::cuda
