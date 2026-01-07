#pragma once

#include "caf/cuda/all.hpp"

// Control-layer object types
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include "caf/cuda/control-layer/memory_transfer_token.hpp"

#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/token_factory.hpp"

// -----------------------------------------------------------------------------
// Type IDs (required for typed behaviors)
// -----------------------------------------------------------------------------

CAF_BEGIN_TYPE_ID_BLOCK(cuda_control, caf::first_custom_type_id + 200)

CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::behavior_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::launch_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::launch_response_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::memory_transfer_token>))

CAF_END_TYPE_ID_BLOCK(cuda_control)

// -----------------------------------------------------------------------------
// Unsafe: explicitly local-only, never serialized
// -----------------------------------------------------------------------------

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::behavior_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_response_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::memory_transfer_token)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::behavior_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::launch_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::launch_response_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::memory_transfer_token>)
