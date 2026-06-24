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

/// GEMV Actor for single-precision matrix-vector multiplication.
/// Message Signature: (in<float> A, in<float> x, out<float> y, int m, int n, [float alpha, float beta])
class gemv_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<gemv_actor>(reply_id);
  }

  gemv_actor(caf::actor_config& cfg, int reply_id = 0) 
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~gemv_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      // Standard host buffer based calls
      [this](in<float> A, in<float> x, out<float> y, int m, int n) {
        enqueue_gemv(-1, actor_id_, A, x, y, m, n, 1.0f, 0.0f, false);
      },
      [this](in<float> A, in<float> x, out<float> y, int m, int n, float alpha, float beta) {
        enqueue_gemv(-1, actor_id_, A, x, y, m, n, alpha, beta, false);
      },
      // Routing control overloads
      [this](int device_num, int stream_id, in<float> A, in<float> x, out<float> y, int m, int n) {
        enqueue_gemv(device_num, stream_id, A, x, y, m, n, 1.0f, 0.0f, false);
      },
      [this](int device_num, int stream_id, in<float> A, in<float> x, out<float> y, int m, int n, float alpha, float beta) {
        enqueue_gemv(device_num, stream_id, A, x, y, m, n, alpha, beta, false);
      },
      // mem_ptr based calls (useful for pipelines)
      [this](mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, int m, int n) {
        enqueue_gemv(-1, actor_id_, A, x, y, m, n, 1.0f, 0.0f, false);
      },
      [this](mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, int m, int n, float alpha, float beta) {
        enqueue_gemv(-1, actor_id_, A, x, y, m, n, alpha, beta, false);
      },
      [this](int device_num, int stream_id, mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, int m, int n) {
        enqueue_gemv(device_num, stream_id, A, x, y, m, n, 1.0f, 0.0f, false);
      },
      [this](int device_num, int stream_id, mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, int m, int n, float alpha, float beta) {
        enqueue_gemv(device_num, stream_id, A, x, y, m, n, alpha, beta, false);
      },
      // Mem ptr return overloads
      [this](return_mem_ptr_atom, in<float> A, in<float> x, out<float> y, int m, int n) {
        enqueue_gemv(-1, actor_id_, A, x, y, m, n, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, in<float> A, in<float> x, out<float> y, int m, int n, float alpha, float beta) {
        enqueue_gemv(-1, actor_id_, A, x, y, m, n, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, in<float> A, in<float> x, out<float> y, int m, int n) {
        enqueue_gemv(device_num, stream_id, A, x, y, m, n, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, in<float> A, in<float> x, out<float> y, int m, int n, float alpha, float beta) {
        enqueue_gemv(device_num, stream_id, A, x, y, m, n, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, int m, int n) {
        enqueue_gemv(-1, actor_id_, A, x, y, m, n, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, int m, int n, float alpha, float beta) {
        enqueue_gemv(-1, actor_id_, A, x, y, m, n, alpha, beta, true);
      },
      [this](return_mem_ptr_atom,int device_num, int stream_id, mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, int m, int n) {
        enqueue_gemv(device_num, stream_id, A, x, y, m, n, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, int m, int n, float alpha, float beta) {
        enqueue_gemv(device_num, stream_id, A, x, y, m, n, alpha, beta, true);
      },
    };
  }

private:
  // Overload for Host-wrapped buffers
  void enqueue_gemv(int device_num, int stream_id, 
                    in<float> A_arg, in<float> x_arg, out<float> y_arg, 
                    int m, int n, float alpha, float beta, bool return_ptrs) {
    command_runner<in<float>, in<float>, out<float>> runner;
    
    // Allocate/Transfer memory.
    auto results = runner.transfer_memory(device_num, stream_id, A_arg, x_arg, y_arg);
    execute_and_reply(device_num, stream_id, std::get<0>(results), 
                      std::get<1>(results), std::get<2>(results), 
                      m, n, alpha, beta, return_ptrs);
  }

  // Overload for already-existing Device buffers
  void enqueue_gemv(int device_num, int stream_id, 
                    mem_ptr<float> A_ptr, mem_ptr<float> x_ptr, mem_ptr<float> y_ptr, 
                    int m, int n, float alpha, float beta, bool return_ptrs) {
    // Pass through command_runner to ensure proper ref-counting/scheduling
    command_runner<mem_ptr<float>, mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, A_ptr, x_ptr, y_ptr);
    execute_and_reply(device_num, stream_id, std::get<0>(results), 
                      std::get<1>(results), std::get<2>(results), 
                      m, n, alpha, beta, return_ptrs);
  }

  void execute_and_reply(int device_num, int stream_id, 
                         mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y, 
                         int m, int n, float alpha, float beta, bool return_ptrs) {
    // Get device based on scheduling rules
    auto plat = platform::create();
    device_ptr dev;
    if (device_num == -1)
      dev = plat->schedule(stream_id);
    else
      dev = plat->schedule(stream_id, device_num);

    // Perform cuBLAS operation
    dev->sgemv(stream_id, m, n, alpha, A, x, beta, y);

    // Handle message routing back to requester
    handle_reply(device_num, stream_id, A, x, y, return_ptrs);
  }

  void handle_reply(int device_num, int stream_id, 
                    mem_ptr<float> A_ptr, mem_ptr<float> x_ptr, mem_ptr<float> y_ptr, 
                    bool return_ptrs) {
    command_runner<mem_ptr<float>> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;
    
    auto r_id = reply_id_;

    if (return_ptrs) {
      caf::anon_mail(r_id, A_ptr, x_ptr, y_ptr).send(sender);
    } else {
      runner.copy_to_host_async(y_ptr, [sender, r_id](std::vector<float>&& data) {
        if (sender) {
          caf::anon_mail(r_id, 2, std::move(data)).send(sender);
        }
      });
    }
  }

  int actor_id_;
  int reply_id_;
};

} // namespace caf::cuda