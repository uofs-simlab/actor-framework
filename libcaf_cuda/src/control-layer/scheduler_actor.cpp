#include "caf/cuda/control-layer/all-control-layer.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/green_light_behavior.hpp"
#include "caf/cuda/control-layer/red_light_behavior.hpp"
#include "caf/cuda/control-layer/core_usage_behavior.hpp"
#include <string>
#include <iostream>

/*
 * This class is meant to handle actor GPU scheduling via s/r/r IPC
 * it has nothing to do with the scheduler class, that is kernel layer
 */
namespace caf::cuda {

caf::behavior scheduler_actor(caf::stateful_actor<scheduler_actor_state>* self, int device_number) {
    auto& state = self->state();

    // add self reference
    state.self = self;

    // set device number
    state.device_number = device_number;

    static red_light_behavior red_behavior(state);
    static green_light_behavior green_behavior(state);
    static core_usage_behavior core_behavior(state);

    // populate the behavior table
    state.table.add("red", &red_behavior);
    state.table.add("green",   &green_behavior);
    state.table.add("core_usage",   &core_behavior);

    // default behavior
    state.current_behavior = state.table.get(behavior_token("green"));
    state.current_behavior->on_enter();

    return {
        [=](const token_ptr& tok) {
            // std::cout << "Received token\n";
            state.current_behavior->receive(tok);
        },
        [&state](const caf::cuda::behavior_token_ptr& tok) -> bool {
    auto* next = state.table.get(*tok);
    if (next) {
        if (next != state.current_behavior) {
            state.current_behavior->on_exit();   // cleanup current behavior
            state.current_behavior = next;       // swap behavior
            state.current_behavior->on_enter();  // init new behavior
            std::cout << "[INFO] Behavior changed to: " << tok->name() << "\n";
            return true; // behavior changed
        } else {
            std::cout << "[INFO] Behavior already active: " << tok->name() << "\n";
            return false; // behavior was already current
        }
    } else {
        std::cout << "[WARN] No behavior found for token: " << tok->name() << "\n";
        return false; // no change
    }
	}
	,
        [=](std::vector<token_ptr> tokens) {
            for (size_t i = 0; i < tokens.size(); ++i) {
                state.current_behavior->receive(tokens[i]);
            }
        },
        [=](std::string word) {
            // std::cout << "Received message " << word << "\n";
        },
        [=](caf::cuda::mem_ptr<int> token) {
            if (!token) {
                std::cout << "Received null mem_ptr\n";
                return;
            }
            if (token->is_scalar()) {
                std::cout << "Received mem_ptr with scalar value: "
                          << *token->host_scalar_ptr() << "\n";
            } else {
                std::cout << "Received mem_ptr with "
                          << token->size() << " elements\n";
                // Optional: print fake data if testing copy_to_host
                // auto host_data = token->copy_to_host();
                // for (auto v : host_data) std::cout << v << " ";
                // std::cout << "\n";
            }
        }
    };
}

} // namespace caf::cuda
