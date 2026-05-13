#pragma once

#include <functional>
#include <queue>

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>
#include <caf/anon_mail.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/response_promise.hpp>

#include <cuda.h>

#include "caf/cuda/command.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/helpers.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/program.hpp"

/*

namespace caf::cuda {

// ---------------------------------------------------------------------------
// actor_facade: Stateless wrapper for GPU kernels. Launches async work via 
// command_runner and notifies the requester directly from CUDA callbacks.
// ---------------------------------------------------------------------------
template <bool PassConfig, class... Ts>
class actor_facade : public event_based_actor {
public:
  using mem_tuple = std::tuple<mem_ptr<raw_t<Ts>>...>;

  static caf::actor create(caf::actor_system& sys,
                           program_ptr program,
                           nd_range dims,
                           atom_value reply_atom = 0) {
    return caf::actor_cast<caf::actor>(
      sys.spawn<actor_facade>(std::move(program), std::move(dims), reply_atom));
  }

  actor_facade(caf::actor_config& cfg, program_ptr program, nd_range dims, atom_value reply_atom)
    : caf::event_based_actor(cfg),
      program_(std::move(program)),
      dims_(std::move(dims)),
      reply_atom_(reply_atom) {
    actor_id_ = this->id();
  }

  ~actor_facade() override {
      command_runner<Ts...> runner;
      runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      [this](atom_value stage, int device_num, Ts... args) {
        enqueue_impl(device_num, stage, std::move(args)...);
      },
      [this](atom_value stage, Ts... args) {
        enqueue_impl(-1, stage, std::move(args)...);
      },
      [this](Ts... args) {
        enqueue_impl(-1, kernel_atom_v, std::move(args)...);
      }
    };
  }

private:
  template <class... Us>
  void enqueue_impl(int device_num, atom_value stage, Us&&... xs) {
    command_runner<Ts...> runner;
    // Pass actor_id_ as stream_id, and the received device_num
    auto results = runner.run_async(program_, dims_, actor_id_, 0, device_num, std::forward<Us>(xs)...);

    auto sender = this->current_sender();
    auto r_atom = reply_atom_;
    auto stream_id = actor_id_;

    atom_value stage_done = gpu_done_atom_v;
    if (stage == htod_atom_v) stage_done = htod_done_atom_v;
    else if (stage == kernel_atom_v) stage_done = kernel_done_atom_v;
    else if (stage == dtoh_atom_v) stage_done = dtoh_done_atom_v;

    // Pass actor_id_ as stream_id, and the received device_num to add_callback
    runner.add_callback(stream_id, device_num, [sender, r_atom, stage_done, results]() mutable {
      auto msg = caf::make_message(stage_done, results);
      if (r_atom != 0) {
        caf::anon_mail(r_atom, std::move(msg)).send(sender);
      } else {
        caf::anon_mail(std::move(msg)).send(sender);
      }
    });
  }

  program_ptr  program_;
  nd_range     dims_;
  caf::actor_id actor_id_;
  atom_value   reply_atom_;
};

} // namespace caf::cuda
 */