#pragma once

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>

#include "caf/cuda/actor_facade.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"

namespace caf::detail {

// Helper to spawn actor_facade.
// Ts... are the types of the kernel arguments (e.g., in<int>, out<int>).
template <bool PassConfig, class... Ts>
struct cuda_spawn_helper {
  // Us... are the actual values of the kernel arguments passed to spawnFromCUBIN.
  // These are NOT passed to the actor_facade constructor/create method.
  // They are sent as a message to the spawned actor_facade later.
  template <class... Us>
  caf::actor operator()(caf::actor_system* sys, caf::actor_config&& /*cfg*/,
                        caf::cuda::program_ptr prog, caf::cuda::nd_range dims,
                        Us&&... /*xs*/) const {
    using impl = caf::cuda::actor_facade<PassConfig, Ts...>;
    return impl::create(*sys, std::move(prog), std::move(dims), 0);
  }
};

} // namespace caf::detail