#pragma once

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>

#include "caf/cuda/actor_facade_event.hpp"

namespace caf::cuda {

template <bool PassConfig, class... Ts>
class actor_facade : public actor_facade_event<PassConfig, Ts...> {
public:
  using super = actor_facade_event<PassConfig, Ts...>;

  static caf::actor create(caf::actor_system& sys,
                           caf::actor_config&&,
                           program_ptr program,
                           nd_range dims,
                           Ts&&...) {
    return caf::actor_cast<caf::actor>(
      sys.spawn<actor_facade>(std::move(program), std::move(dims)));
  }

  static caf::actor create(caf::actor_system* sys,
                           caf::actor_config&&,
                           program_ptr program,
                           nd_range dims,
                           Ts&&...) {
    return caf::actor_cast<caf::actor>(
      sys->spawn<actor_facade>(std::move(program), std::move(dims)));
  }

  actor_facade(caf::actor_config& cfg, program_ptr prog, nd_range dims)
    : super(cfg, std::move(prog), std::move(dims)) {
  }
};

} // namespace caf::cuda
