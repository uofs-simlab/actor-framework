#pragma once

#include "caf/cuda/all.hpp"
#include "caf/cuda/global.hpp"




// Control-layer object types
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/response_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include "caf/cuda/control-layer/memory_transfer_token.hpp"
#include "caf/cuda/control-layer/memory_response_token.hpp"
#include "caf/cuda/control-layer/transfer_token.hpp"

//scheduler actor includes
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/token_factory.hpp"

//memory actor includes
#include "caf/cuda/control-layer/memory_actor/memory_actor.hpp"
#include "caf/cuda/control-layer/memory_actor/memory_request_token.hpp"
#include "caf/cuda/control-layer/memory_actor/mem_token.hpp"

//exit actor includes
#include "caf/cuda/control-layer/exit_actor/exit_actor.hpp"


// -----------------------------------------------------------------------------
// Type IDs (required for typed behaviors)
// -----------------------------------------------------------------------------

CAF_BEGIN_TYPE_ID_BLOCK(cuda_control, caf::first_custom_type_id + 12000)

CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::behavior_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::launch_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::launch_response_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::response_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::memory_transfer_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::memory_response_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::cuda::memory_request_token))
CAF_ADD_TYPE_ID(cuda_control, (caf::cuda::mem_token))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::transfer_token>))
CAF_ADD_TYPE_ID(cuda_control, (std::vector<caf::intrusive_ptr<caf::cuda::token>>))

CAF_END_TYPE_ID_BLOCK(cuda_control)

// -----------------------------------------------------------------------------
// Unsafe: explicitly local-only, never serialized
// -----------------------------------------------------------------------------

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::behavior_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_response_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::memory_transfer_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::memory_request_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::mem_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::memory_response_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::response_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::transfer_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::behavior_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::launch_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::launch_response_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::response_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::transfer_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::memory_transfer_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::memory_response_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<caf::intrusive_ptr<caf::cuda::token>>)


