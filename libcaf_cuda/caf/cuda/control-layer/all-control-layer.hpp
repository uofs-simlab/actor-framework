#pragma once

#include "caf/cuda/all.hpp"

// Control-layer object types (ref-counted, non-copyable)
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"

// Control-layer logic
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/token_factory.hpp"

// -----------------------------------------------------------------------------
// Type ID block
//   Register ONLY the types that may appear by value in caf::message.
//   Ref-counted control objects are passed exclusively via intrusive_ptr.
// -----------------------------------------------------------------------------

CAF_BEGIN_TYPE_ID_BLOCK(cuda_control, caf::first_custom_type_id + 200)

CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::behavior_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::launch_token>))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::launch_response_token>))

CAF_END_TYPE_ID_BLOCK(cuda_control)

// -----------------------------------------------------------------------------
// Unsafe message types
//   These are explicitly local-only, non-serializable control objects.
//   CAF will reject any attempt to send them over the network.
// -----------------------------------------------------------------------------

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::behavior_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_response_token)

