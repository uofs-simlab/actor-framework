#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/red_light_behavior.hpp"
#include "caf/cuda/control-layer/token_factory.hpp"

namespace caf::cuda {

    void red_light_behavior::schedule() {
    }

    void red_light_behavior::receive(scheduler_actor_state* state, const token_ptr& tok) {
        state->queue.push(tok); // enqueue everything
    }

    red_light_behavior::~red_light_behavior() noexcept = default;
    
    void red_light_behavior::init(scheduler_actor_state * state) {
	   std::cout << "RED LIGHT\n";
	   behavior_token_ptr green_light = make_behavior_token("green");

	   //send a request to change behavior to green light after 5 seconds
    	   anon_mail(green_light)
		   .delay(std::chrono::seconds(5))
		   .send(state -> self);
    }

} // namespace caf::cuda
