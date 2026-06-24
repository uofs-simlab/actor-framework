#pragma once

#include <caf/all.hpp>
#include <queue>
#include "caf/cuda/device.hpp"
#include "caf/cuda/global_export.hpp"
#include "caf/cuda/control-layer/memory_actor/memory_request_token.hpp"
#include "caf/cuda/control-layer/memory_actor/mem_token.hpp"
/*
 * The memory actor 
 * meant to help prevent out of memory errors by employing s/r/r IPC 
 * it will reply to a request once it believes there is enough memory to serve it
 * as of right now still is a best effort service as due to fragmentation
 * replying still may result in an out of memory error
 */

namespace caf::cuda {

struct memory_actor_state {

	std::vector<std::queue<memory_request_token>> requests;
	std::vector<device_ptr> devices;
	std::vector<std::size_t> available_memory;
	int num_devices;
	bool pending_requests = false;

};

caf::behavior CAF_CUDA_EXPORT memory_actor(caf::stateful_actor<memory_actor_state> * self,
		int num_devices);
}

