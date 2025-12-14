#pragma once
#include "caf/cuda/all.hpp"
#include "caf/cuda/control-layer/token.hpp"
#include "caf/cuda/control-layer/launch_token.hpp"
#include "caf/cuda/control-layer/launch_response_token.hpp"
#include "caf/cuda/control-layer/behavior_token.hpp"
#include "caf/cuda/control-layer/behavior.hpp"
#include "caf/cuda/control-layer/scheduler_actor.hpp"
#include "caf/cuda/control-layer/token_factory.hpp"

CAF_BEGIN_TYPE_ID_BLOCK(cuda_control, caf::first_custom_type_id + 200)
// You can also use  caf::id_block::core::end  instead of the +200 if you prefer

  CAF_ADD_TYPE_ID(cuda_control, (caf::cuda::token))
CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::token>))  
//CAF_ADD_TYPE_ID(cuda_control, (caf::cuda::behavior_token))
  CAF_ADD_TYPE_ID(cuda_control, (caf::intrusive_ptr<caf::cuda::behavior_token>)) // ← and this one too if you ever use it
	//CAF_ADD_TYPE_ID(cuda_control, (caf::cuda::launch_token))
  //CAF_ADD_TYPE_ID(cuda_control, (caf::cuda::launch_response_token))

  // Very commonly needed as well – add them now so you don’t hit the error later

CAF_END_TYPE_ID_BLOCK(cuda_control)

// Optional but harmless – keep your old macros (they silence the “unsafe” warning)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::token_ptr)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::behavior_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::behavior_token_ptr)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_token)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(caf::cuda::launch_response_token)
