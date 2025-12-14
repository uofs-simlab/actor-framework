#pragma once

#include "caf/cuda/all.hpp"

// Control-layer object types (local-only, ref-counted)
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"

// Control-layer logic
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/token_factory.hpp"

// -----------------------------------------------------------------------------
// NO TYPE ID BLOCK
//
// These control-layer types are:
//   - local-only
//   - never serialized
//   - never sent over the network
//   - ref-counted and non-copyable
//
// Therefore they must NOT be registered.
// -----------------------------------------------------------------------------

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::behavior_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_response_token)

// And most importantly:
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::behavior_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::launch_token>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::intrusive_ptr<caf::cuda::launch_response_token>)

