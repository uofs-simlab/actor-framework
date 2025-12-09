#pragma once
#include <caf/all.hpp>
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include <queue>

/*
 * The scheduler actor 
 * meant to make scheduling decisions using s/r/r ipc
 */

struct scheduler_actor_state {
	std::queue<token_ptr> queue;
}


caf::behavior scheduler_actor(caf::stateful_actor<scheduler_actor_state>);

