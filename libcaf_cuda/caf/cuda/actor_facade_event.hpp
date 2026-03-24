#pragma once

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>
#include <caf/error.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/make_counted.hpp>
#include <caf/message.hpp>
#include <caf/result.hpp>
#include <caf/sec.hpp>

#include "caf/cuda/command.hpp"
#include "caf/cuda/helpers.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/program.hpp"

namespace caf::cuda {

template <bool PassConfig, class... Ts>
class actor_facade_event : public caf::event_based_actor {
public:
  static caf::actor create(caf::actor_system& sys,
                           caf::actor_config&&,
                           program_ptr program,
                           nd_range dims,
                           Ts&&...) {
    return caf::actor_cast<caf::actor>(
      sys.spawn<actor_facade_event>(std::move(program), std::move(dims)));
  }

  static caf::actor create(caf::actor_system* sys,
                           caf::actor_config&&,
                           program_ptr program,
                           nd_range dims,
                           Ts&&...) {
    return caf::actor_cast<caf::actor>(
      sys->spawn<actor_facade_event>(std::move(program), std::move(dims)));
  }

  actor_facade_event(caf::actor_config& cfg, program_ptr program, nd_range dims)
    : caf::event_based_actor(cfg),
      program_(std::move(program)),
      dims_(std::move(dims)),
      actor_id_(random_number()) {
  }

  ~actor_facade_event() override {
    platform::create()->release_streams_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      [this](const caf::message& msg) -> caf::result<std::vector<output_buffer>> {
        try {
          if (auto outputs = handle_message(msg))
            return std::move(*outputs);
          return caf::make_error(caf::sec::unexpected_message,
                                 "actor_facade_event received unsupported arguments");
        } catch (const std::exception& ex) {
          return caf::make_error(caf::sec::runtime_error, ex.what());
        }
      },
    };
  }

protected:
  template <class... Us>
  std::vector<output_buffer> launch_kernel(Us&&... xs) {
    using command_t = command<caf::actor, std::decay_t<Us>...>;
    auto cmd = caf::make_counted<command_t>(program_, dims_, actor_id_,
                                            std::forward<Us>(xs)...);
    return cmd->enqueue();
  }

private:
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
  nd_range dims_;
  int actor_id_;
};

} // namespace caf::cuda