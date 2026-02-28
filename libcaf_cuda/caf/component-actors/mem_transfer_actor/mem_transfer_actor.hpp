#pragma once

#include <caf/all.hpp>
#include "caf/cuda/mem_ref.hpp"

/*
 * This actor is meant to handle memory transfer requests 
 * since copy_to_host is blocking we need this actor to 
 * block while not blocking the scheduler thread 
 */


namespace caf::cuda {

template <class T>
void mem_transfer_actor_fun(caf::blocking_actor* self) {
  self->receive_while([&] { return true; })(
    [&](mem_ptr<T> d_mem) {

      // Blocking call happens safely here
      std::vector<T> host_data = d_mem->copy_to_host();

      self->reply(std::move(host_data));
    }
  );
}

} // namespace caf::cuda
