#pragma once

#include <functional>
#include <queue>

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>
#include <caf/anon_mail.hpp>
#include <caf/event_based_actor.hpp>
#include <cuda.h>

#include "caf/cuda/command.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/helpers.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/command_runner.hpp"


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
                           int reply_id = 0) {
    return caf::actor_cast<caf::actor>(
      sys.spawn<actor_facade>(std::move(program), std::move(dims), reply_id));
  }

  actor_facade(caf::actor_config& cfg, program_ptr program, nd_range dims, int reply_id)
    : caf::event_based_actor(cfg),
      program_(std::move(program)),
      dims_(std::move(dims)),
      reply_id_(reply_id) {
    actor_id_ = this->id();
  }

  ~actor_facade() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      [this](int device_num, int stream_id, std::vector<int> output_indices, Ts... args) {
        enqueue_impl(device_num, stream_id, std::move(output_indices), std::forward<Ts>(args)...);
      },
      [this](int device_num, int stream_id, Ts... args) {
        enqueue_impl(device_num, stream_id, {}, std::forward<Ts>(args)...);
      },
      [this](int device_num, std::vector<int> output_indices, Ts... args) {
        enqueue_impl(device_num, static_cast<int>(actor_id_), std::move(output_indices),
                     std::forward<Ts>(args)...);
      },
      [this](std::vector<int> output_indices, Ts... args) {
        enqueue_impl(-1, static_cast<int>(actor_id_), std::move(output_indices),
                     std::forward<Ts>(args)...);
      },
      [this](int device_num, Ts... args) {
        // Copy everything back if indices are omitted
        enqueue_impl(device_num, static_cast<int>(actor_id_), {}, std::forward<Ts>(args)...);
      },
      [this](Ts... args) {
        // Copy everything back if indices are omitted
        enqueue_impl(-1, static_cast<int>(actor_id_), {}, std::forward<Ts>(args)...);
      }
    };
  }

private:
  template <class... Us>
  void enqueue_impl(int device_num, int stream_id, std::vector<int> output_indices, Us&&... xs) {
    command_runner<Ts...> runner;
    auto results = runner.run_async(program_, dims_, stream_id, 0, 
                                    device_num, std::forward<Us>(xs)...);

    auto sender = this->current_sender();
    auto r_id = reply_id_;

    if (output_indices.empty()) {
      for (int i = 0; i < static_cast<int>(sizeof...(Ts)); ++i) {
        output_indices.push_back(i);
      }
    }

    for (int idx : output_indices) {
      if (idx >= 0 && idx < static_cast<int>(sizeof...(Ts))) {
        // Dispatch runtime index to compile-time sequence
        dispatch_index(idx, [&](auto current_idx_constant) {
          constexpr std::size_t Index = current_idx_constant;
          using MemPtrType = std::tuple_element_t<Index, mem_tuple>;
          using ValueType = typename MemPtrType::element_type::value_type;

          MemPtrType mem_ptr = std::get<Index>(results);

          if (mem_ptr && (mem_ptr->access() == OUT || mem_ptr->access() == IN_OUT)) {
            runner.copy_to_host_async(mem_ptr, [sender, r_id, Index](std::vector<ValueType>&& data) {
              if (sender) {
                // Send: Correlation ID, Argument Index, Data Vector
                caf::anon_mail(r_id, static_cast<int>(Index), std::move(data)).send(sender);
              }
            });
          }
        });
      } else {
        this->println("Warning: Output index {} is out of bounds", idx);
      }
    }

    // Final completion alert (Correlation ID, -1 to signal "all finished")
    runner.add_callback(stream_id, device_num, [sender, r_id]() mutable {
      if (sender) {
        caf::anon_mail(r_id, -1).send(sender);
      }
    });
  }

  // Helper to map runtime index to compile-time index for tuple access
  template <class F>
  void dispatch_index(int idx, F&& f) {
    dispatch_index_helper(std::make_index_sequence<sizeof...(Ts)>{}, idx, std::forward<F>(f));
  }

  template <std::size_t... Is, class F>
  void dispatch_index_helper(std::index_sequence<Is...>, int idx, F&& f) {
    (..., (static_cast<int>(Is) == idx ? f(std::integral_constant<std::size_t, Is>{}) : (void)0));
  }

  program_ptr  program_;
  nd_range     dims_;
  caf::actor_id actor_id_;
  int          reply_id_;
};

} // namespace caf::cuda