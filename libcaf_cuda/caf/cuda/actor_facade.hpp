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
#include <utility> // For std::index_sequence, std::integral_constant

#include "caf/cuda/command.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/helpers.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/program.hpp"


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
      [this](int device_num, int stream_id, std::vector<int> output_indices, Ts... args) {
        enqueue_impl(device_num, stream_id, std::move(output_indices), std::forward<Ts>(args)...);
      },
      [this](int device_num, int stream_id, Ts... args) {
        // Copy everything back if indices are omitted
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
    // Launch kernel asynchronously
    auto results = runner.run_async(program_, dims_, stream_id, 0, 
                                    device_num, std::forward<Us>(xs)...);

    auto sender = this->current_sender();
    auto r_atom = reply_atom_;

    // If no indices specified, default to copying all arguments back
    if (output_indices.empty()) {
      for (int i = 0; i < static_cast<int>(sizeof...(Ts)); ++i) {
        output_indices.push_back(i);
      }
    }

    // Track if we actually queued any transfers to host
    // Even if we don't, we will send a final 'done' signal via callback
    // to ensure the stream is drained.

    if (!output_indices.empty()) {
        // Helper to dispatch a lambda based on a runtime index
        auto switch_on_index = [&](int runtime_idx, auto&& func, auto... Is) {
            ([&] {
                if (runtime_idx == Is) {
                    func(std::integral_constant<int, Is>{});
                }
            }(), ...);
        };

        for (int idx : output_indices) {
            if (idx >= 0 && idx < sizeof...(Ts)) {
                switch_on_index(idx, [&](auto current_idx_constant) {
                    // Compile-time index extraction
                    constexpr std::size_t Index = current_idx_constant; // Compile-time index
                    using MemPtrType = std::tuple_element_t<Index, mem_tuple>;
                    using ValueType = typename MemPtrType::element_type::value_type;

                    MemPtrType mem_ptr = std::get<Index>(results);

                    // Only copy if the mem_ptr is valid and has OUT or IN_OUT access
                    if (mem_ptr && (mem_ptr->access() == OUT || mem_ptr->access() == IN_OUT)) {
                        runner.copy_to_host_async(mem_ptr, 
                          [sender, r_atom, Index](std::vector<ValueType>&& data) {
                            // Send back the data along with the original index
                            if (sender) { // Ensure sender is still valid
                                if (r_atom != 0) {
                                    caf::anon_mail(r_atom, Index, std::move(data)).send(sender);
                                } else {
                                    caf::anon_mail(Index, std::move(data)).send(sender);
                                }
                            }
                        });
                    }
                }, std::make_index_sequence<sizeof...(Ts)>{});
            } else {
                this->println("Warning: Output index {} is out of bounds (0-{})", idx, sizeof...(Ts) - 1);
            }
        }
    }

    // Finally, alert the sender that everything is finished.
    // This callback is queued on the stream after all kernel and transfer commands.
    runner.add_callback(stream_id, device_num, [sender, r_atom]() mutable {
      if (sender) {
        if (r_atom != 0)
          caf::anon_mail(r_atom, gpu_done_atom_v).send(sender);
        else
          caf::anon_mail(gpu_done_atom_v).send(sender);
      }
    });
  }

  program_ptr  program_;
  nd_range     dims_;
  caf::actor_id actor_id_;
  atom_value   reply_atom_;
};

} // namespace caf::cuda