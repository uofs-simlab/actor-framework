#pragma once

#include <caf/all.hpp>
#include "caf/cuda/mem_ref.hpp"
#include "caf/cuda/control-layer/all-control-layer.hpp"

/*
 * This actor is meant to act as a callback actor that will synchronize with a 
 * stream bound to a mem_ref and reply with when it is awake 
 * it can also synchronize and release the response token when it awakes for the 
 * scheduler actor 
 * giving the kernels as events illusion for the scheduler actor 
 */


namespace caf::cuda {

template <class T>
void sync_actor_fun(caf::blocking_actor* self) {

  self->receive(
    [&](mem_ptr<T> d_mem) -> mem_ptr<T> {

      // blocking call is safe here
      d_mem -> synchronize();  
    return d_mem;
    },
    [&](mem_ptr<T> d_mem, response_token_ptr res) -> mem_ptr<T> {
      d_mem->synchronize();

      res ->release();
       
      return d_mem;
    }
  );
}

} // namespace caf::cuda
