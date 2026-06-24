#pragma once

#include <caf/actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/actor_config.hpp>
#include <caf/actor_system.hpp>
#include <caf/anon_mail.hpp>
#include <caf/event_based_actor.hpp>

#include "caf/cuda/device.hpp"
#include "caf/cuda/mem_ref.hpp"
#include "caf/cuda/command_runner.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/types.hpp"

namespace caf::cuda {

/// AXPY Actor for single-precision vector-vector addition.
/// Message Signature: (in<float> x, in_out<float> y, int n, float alpha)
class axpy_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<axpy_actor>(reply_id);
  }

  axpy_actor(caf::actor_config& cfg, int reply_id = 0) 
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~axpy_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      // Standard host buffer based calls
      [this](in<float> x, in_out<float> y, int n, float alpha) {
        enqueue_axpy(-1, actor_id_, x, y, n, alpha, false);
      },
      // Routing control overloads
      [this](int device_num, int stream_id, in<float> x, in_out<float> y, int n, float alpha) {
        enqueue_axpy(device_num, stream_id, x, y, n, alpha, false);
      },
      // mem_ptr based calls (useful for pipelines)
      [this](mem_ptr<float> x, mem_ptr<float> y, int n, float alpha) {
        enqueue_axpy(-1, actor_id_, x, y, n, alpha, false);
      },
      [this](int device_num, int stream_id, mem_ptr<float> x, mem_ptr<float> y, int n, float alpha) {
        enqueue_axpy(device_num, stream_id, x, y, n, alpha, false);
      },
      // Mem ptr return overloads
      [this](return_mem_ptr_atom, in<float> x, in_out<float> y, int n, float alpha) {
        enqueue_axpy(-1, actor_id_, x, y, n, alpha, true);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, in<float> x, in_out<float> y, int n, float alpha) {
        enqueue_axpy(device_num, stream_id, x, y, n, alpha, true);
      },
      [this](return_mem_ptr_atom, mem_ptr<float> x, mem_ptr<float> y, int n, float alpha) {
        enqueue_axpy(-1, actor_id_, x, y, n, alpha, true);
      }, [this](return_mem_ptr_atom,int device_num, int stream_id,mem_ptr<float> x, mem_ptr<float> y, int n, float alpha) {
        enqueue_axpy(device_num, stream_id, x, y, n, alpha, true);
      },
    };
  }

private:
  // Overload for Host-wrapped buffers
  void enqueue_axpy(int device_num, int stream_id, 
                    in<float> x_arg, in_out<float> y_arg, 
                    int n, float alpha, bool return_ptrs) {
    command_runner<in<float>, in_out<float>> runner;
    
    // Allocate/Transfer memory.
    auto results = runner.transfer_memory(device_num, stream_id, x_arg, y_arg);
    execute_and_reply(device_num, stream_id, std::get<0>(results), 
                      std::get<1>(results), n, alpha, return_ptrs);
  }

  // Overload for already-existing Device buffers
  void enqueue_axpy(int device_num, int stream_id, 
                    mem_ptr<float> x_ptr, mem_ptr<float> y_ptr, 
                    int n, float alpha, bool return_ptrs) {
    // Pass through command_runner to ensure proper ref-counting/scheduling
    command_runner<mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, x_ptr, y_ptr);
    execute_and_reply(device_num, stream_id, std::get<0>(results), 
                      std::get<1>(results), n, alpha, return_ptrs);
  }

  void execute_and_reply(int device_num, int stream_id, 
                         mem_ptr<float> x, mem_ptr<float> y, 
                         int n, float alpha, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev;
    if (device_num == -1)
      dev = plat->schedule(stream_id);
    else
      dev = plat->schedule(stream_id, device_num);

    // Perform cuBLAS operation
    dev->saxpy(stream_id, n, alpha, x, y);

    handle_reply(device_num, stream_id, x, y, return_ptrs);
  }

  void handle_reply(int, int, mem_ptr<float> x_ptr, mem_ptr<float> y_ptr, bool return_ptrs) {
    command_runner<mem_ptr<float>> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;
    
    auto r_id = reply_id_;

    if (return_ptrs) {
      caf::anon_mail(r_id, x_ptr, y_ptr).send(sender);
    } else {
      runner.copy_to_host_async(y_ptr, [sender, r_id](std::vector<float>&& data) {
        if (sender) {
          caf::anon_mail(r_id, 1, std::move(data)).send(sender);
        }
      });
    }
  }

  int actor_id_;
  int reply_id_;
};

} // namespace caf::cuda