#pragma once

#include "caf/cuda/global_export.hpp" //here to export files
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include "caf/cuda/control-layer/memory_transfer_token.hpp"
#include "caf/cuda/control-layer/memory_response_token.hpp"
#include <caf/actor.hpp>
#include <string>
#include "caf/cuda/mem_ref.hpp"

namespace caf::cuda {

/// Creates a launch_token (used by users when submitting kernels)
CAF_CUDA_EXPORT token_ptr make_launch_token(program_ptr prog,
                            nd_range range,
                            int memory_usage,
                            std::string id,
                            actor reply_to,
			    int dependency = INDEPENDENT);

/// Creates a launch_response_token (created internally by the scheduler
/// when it accepts a kernel launch request)
CAF_CUDA_EXPORT response_token_ptr make_launch_response_token(
    actor receiver,
    const launch_token& orig,
    int device_number,
    int stream_id,
    int reclaim_value = 0,      // optional
    int reclaim_runtime = 0     // optional
);


/// Creates a behavior_token (special — returns its own strong ptr type)
CAF_CUDA_EXPORT behavior_token_ptr make_behavior_token(std::string name);

//creats a memory transfer token
CAF_CUDA_EXPORT token_ptr make_memory_token(int size,
	       	int direction,
		caf::actor replyActor,
		int dependency = INDEPENDENT);

CAF_CUDA_EXPORT response_token_ptr make_memory_response_token(actor receiver,
                                     const memory_transfer_token& orig,
				     int device_number,
				     int stream_id);



/// Creates a transfer_token from a launch_token
CAF_CUDA_EXPORT response_token_ptr make_transfer_token(
    caf::actor receiver,
    const launch_token& orig,
    int device_number,
    int stream_id
);


//do not use this, for testing only 
CAF_CUDA_EXPORT mem_ptr<int> make_mem_ptr(size_t num_elements);




} // namespace caf::cuda
