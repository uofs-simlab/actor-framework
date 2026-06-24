#pragma once
#include <caf/all.hpp>

namespace caf::cuda {
struct exit_actor_state {
        int completed = 0;
};

caf::behavior CAF_CUDA_EXPORT  exit_actor_fun(caf::stateful_actor<exit_actor_state>* self,int limit); 
} //namespace caf::cuda
