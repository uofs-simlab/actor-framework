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

/// SYRK Actor for single-precision symmetric rank-k update.
/// Formula: C = alpha * A * A^T + beta * C
/// Message Signature: (in<float> A, in_out<float> C, int n, int k, [float alpha, float beta])
class syrk_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<syrk_actor>(reply_id);
  }

  syrk_actor(caf::actor_config& cfg, int reply_id = 0) 
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~syrk_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      // Standard host buffer based calls
      [this](in<float> A, in_out<float> C, int n, int k) {
        enqueue_syrk(-1, actor_id_, A, C, n, k, 1.0f, 0.0f, false);
      },
      [this](in<float> A, in_out<float> C, int n, int k, float alpha, float beta) {
        enqueue_syrk(-1, actor_id_, A, C, n, k, alpha, beta, false);
      },
      // Routing control overloads
      [this](int device_num, int stream_id, in<float> A, in_out<float> C, int n, int k) {
        enqueue_syrk(device_num, stream_id, A, C, n, k, 1.0f, 0.0f, false);
      },
      [this](int device_num, int stream_id, in<float> A, in_out<float> C, int n, int k, float alpha, float beta) {
        enqueue_syrk(device_num, stream_id, A, C, n, k, alpha, beta, false);
      },
      // mem_ptr based calls (Implicit routing)
      [this](mem_ptr<float> A, mem_ptr<float> C, int n, int k) {
        enqueue_syrk(-1, actor_id_, A, C, n, k, 1.0f, 0.0f, false);
      },
      [this](mem_ptr<float> A, mem_ptr<float> C, int n, int k, float alpha, float beta) {
        enqueue_syrk(-1, actor_id_, A, C, n, k, alpha, beta, false);
      },
      // mem_ptr based calls (Explicit routing)
      [this](int device_num, int stream_id, mem_ptr<float> A, mem_ptr<float> C, int n, int k) {
        enqueue_syrk(device_num, stream_id, A, C, n, k, 1.0f, 0.0f, false);
      },
      [this](int device_num, int stream_id, mem_ptr<float> A, mem_ptr<float> C, int n, int k, float alpha, float beta) {
        enqueue_syrk(device_num, stream_id, A, C, n, k, alpha, beta, false);
      },
      // Return mem_ptr with explicit routing
      [this](return_mem_ptr_atom, int device_num, int stream_id, mem_ptr<float> A, mem_ptr<float> C, int n, int k, float alpha, float beta) {
        enqueue_syrk(device_num, stream_id, A, C, n, k, alpha, beta, true);
      },
      // Mem ptr return overloads
      [this](return_mem_ptr_atom, in<float> A, in_out<float> C, int n, int k) {
        enqueue_syrk(-1, actor_id_, A, C, n, k, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, in<float> A, in_out<float> C, int n, int k, float alpha, float beta) {
        enqueue_syrk(-1, actor_id_, A, C, n, k, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, mem_ptr<float> A, mem_ptr<float> C, int n, int k) {
        enqueue_syrk(-1, actor_id_, A, C, n, k, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, mem_ptr<float> A, mem_ptr<float> C, int n, int k, float alpha, float beta) {
        enqueue_syrk(-1, actor_id_, A, C, n, k, alpha, beta, true);
      }
    };
  }

private:
  // Overload for Host-wrapped buffers
  void enqueue_syrk(int device_num, int stream_id, 
                    in<float> A_arg, in_out<float> C_arg, 
                    int n, int k, float alpha, float beta, bool return_ptrs) {
    command_runner<in<float>, in_out<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, A_arg, C_arg);
    execute_and_reply(device_num, stream_id, std::get<0>(results), 
                      std::get<1>(results), n, k, alpha, beta, return_ptrs);
  }

  // Overload for already-existing Device buffers
  void enqueue_syrk(int device_num, int stream_id, 
                    mem_ptr<float> A_ptr, mem_ptr<float> C_ptr, 
                    int n, int k, float alpha, float beta, bool return_ptrs) {
    command_runner<mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, A_ptr, C_ptr);
    execute_and_reply(device_num, stream_id, std::get<0>(results), 
                      std::get<1>(results), n, k, alpha, beta, return_ptrs);
  }

  void execute_and_reply(int device_num, int stream_id, 
                         mem_ptr<float> A, mem_ptr<float> C, 
                         int n, int k, float alpha, float beta, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev;
    if (device_num == -1)
      dev = plat->schedule(stream_id);
    else
      dev = plat->schedule(stream_id, device_num);

    dev->ssyrk(stream_id, n, k, alpha, A, beta, C);

    handle_reply(device_num, stream_id, A, C, return_ptrs);
  }

  void handle_reply(int device_num, int stream_id, 
                    mem_ptr<float> A_ptr, mem_ptr<float> C_ptr, 
                    bool return_ptrs) {
    command_runner<float> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;
    
    auto r_id = reply_id_;

    if (return_ptrs) {
      caf::anon_mail(r_id, A_ptr, C_ptr).send(sender);
    } else {
      // Copy C back to host (index 1 in the original arg list)
      runner.copy_to_host_async(C_ptr, [sender, r_id](std::vector<float>&& data) {
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