#pragma once

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>
#include <caf/event_based_actor.hpp>

#include "caf/cuda/command.hpp"
#include "caf/cuda/helpers.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/program.hpp"


namespace caf::cuda {

template <bool PassConfig, class... Ts>
class actor_facade : public event_based_actor {
public:
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

  actor_facade(caf::actor_config& cfg, program_ptr program, nd_range dims)
    : caf::event_based_actor(cfg),
      program_(std::move(program)),
      dims_(std::move(dims)) {
    actor_id_ = static_cast<int>(this->id());
    platform_ = program_->get_platform();
  }

  ~actor_facade() override {
    if (platform_)
      platform_->release_streams_for_actor(this->id());
  }

  caf::behavior make_behavior() override {
    return {
      [this](const caf::message& msg) -> caf::result<std::vector<output_buffer>> {
        try {
          if (auto outputs = handle_message(msg))
            return std::move(*outputs);
          return caf::make_error(caf::sec::unexpected_message,
                                 "actor_facade received unsupported arguments");
        } catch (const std::exception& ex) {
          return caf::make_error(caf::sec::runtime_error, ex.what());
        }
      },
    };
  }

private:
  template <class... Us>
  std::vector<output_buffer> launch_kernel(Us&&... xs) {
    using command_t = command<caf::actor, std::decay_t<Us>...>;
    auto cmd = caf::make_counted<command_t>(program_, dims_, actor_id_,
                                            std::forward<Us>(xs)...);
    return cmd->enqueue();
  }
  

  template <std::size_t... Is>
  std::vector<output_buffer>
  launch_wrapped_message(const caf::message& msg, size_t offset,
                         std::index_sequence<Is...>) {
    return launch_kernel(msg.get_as<Ts>(Is + offset)...);
  }

  template <std::size_t... Is>
  std::vector<output_buffer>
  launch_raw_message(const caf::message& msg, size_t offset,
                     std::index_sequence<Is...>) {
    return launch_kernel(Ts{msg.get_as<raw_t<Ts>>(Is + offset)}...);
  }


  std::optional<std::vector<output_buffer>> handle_message(const caf::message& msg) {
    auto indices = std::index_sequence_for<Ts...>{};
    if (msg.match_elements<Ts...>())
      return launch_wrapped_message(msg, 0, indices);
    if (msg.match_elements<raw_t<Ts>...>())
      return launch_raw_message(msg, 0, indices);
    if (msg.match_elements<caf::actor, Ts...>())
      return launch_wrapped_message(msg, 1, indices);
    if (msg.match_elements<caf::actor, raw_t<Ts>...>())
      return launch_raw_message(msg, 1, indices);
    return std::nullopt;
  }

  program_ptr program_;
  platform_ptr platform_;
  nd_range dims_;
  int actor_id_;
};

} // namespace caf::cuda
