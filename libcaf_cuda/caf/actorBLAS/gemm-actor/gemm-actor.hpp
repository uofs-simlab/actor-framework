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

/// GEMM Actor for matrix-matrix multiplication.
/// Message Signature: (in<T> A, in<T> B, out<T> C, int m, int n, int k, [T alpha, T beta])
template <typename T>
class gemm_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<gemm_actor<T>>(reply_id);
  }

  gemm_actor(caf::actor_config& cfg, int reply_id = 0)
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~gemm_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      // Standard host buffer based calls (beta = 0.0)
      [this](in<T> A, in<T> B, out<T> C, int m, int n, int k) {
        enqueue_gemm(-1, actor_id_, A, B, C, m, n, k, static_cast<T>(1.0), static_cast<T>(0.0), false);
      },
      // Standard host buffer based calls (explicit alpha, beta)
      [this](in<T> A, in<T> B, in_out<T> C, int m, int n, int k, T alpha, T beta) {
        enqueue_gemm(-1, actor_id_, A, B, C, m, n, k, alpha, beta, false);
      },
      // Routing control overloads (beta = 0.0)
      [this](int device_num, int stream_id, in<T> A, in<T> B, out<T> C, int m, int n, int k) {
        enqueue_gemm(device_num, stream_id, A, B, C, m, n, k, static_cast<T>(1.0), static_cast<T>(0.0), false);
      },
      // Routing control overloads (explicit alpha, beta)
      [this](int device_num, int stream_id, in<T> A, in<T> B, in_out<T> C, int m, int n, int k, T alpha, T beta) {
        enqueue_gemm(device_num, stream_id, A, B, C, m, n, k, alpha, beta, false);
      },
      // mem_ptr based calls (useful for pipelines, beta = 0.0)
      [this](mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k) {
        enqueue_gemm(-1, actor_id_, A, B, C, m, n, k, static_cast<T>(1.0), static_cast<T>(0.0), false);
      },
      // mem_ptr based calls (explicit alpha, beta)
      [this](mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k, T alpha, T beta) {
        enqueue_gemm(-1, actor_id_, A, B, C, m, n, k, alpha, beta, false);
      },
      // mem_ptr based calls with routing (beta = 0.0)
      [this](int device_num, int stream_id, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k) {
        enqueue_gemm(device_num, stream_id, A, B, C, m, n, k, static_cast<T>(1.0), static_cast<T>(0.0), false);
      },
      // mem_ptr based calls with routing (explicit alpha, beta)
      [this](int device_num, int stream_id, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k, T alpha, T beta) {
        enqueue_gemm(device_num, stream_id, A, B, C, m, n, k, alpha, beta, false);
      },
      // Mem ptr return overloads (beta = 0.0)
      [this](return_mem_ptr_atom, in<T> A, in<T> B, out<T> C, int m, int n, int k) {
        enqueue_gemm(-1, actor_id_, A, B, C, m, n, k, static_cast<T>(1.0), static_cast<T>(0.0), true);
      },
      // Mem ptr return overloads (explicit alpha, beta)
      [this](return_mem_ptr_atom, in<T> A, in<T> B, in_out<T> C, int m, int n, int k, T alpha, T beta) {
        enqueue_gemm(-1, actor_id_, A, B, C, m, n, k, alpha, beta, true);
      },
      // Mem ptr return overloads with routing (beta = 0.0)
      [this](return_mem_ptr_atom, int device_num, int stream_id, in<T> A, in<T> B, out<T> C, int m, int n, int k) {
        enqueue_gemm(device_num, stream_id, A, B, C, m, n, k, static_cast<T>(1.0), static_cast<T>(0.0), true);
      },
      // Mem ptr return overloads with routing (explicit alpha, beta)
      [this](return_mem_ptr_atom, int device_num, int stream_id, in<T> A, in<T> B, in_out<T> C, int m, int n, int k, T alpha, T beta) {
        enqueue_gemm(device_num, stream_id, A, B, C, m, n, k, alpha, beta, true);
      },
      // Mem ptr return overloads for already-existing Device buffers (beta = 0.0)
      [this](return_mem_ptr_atom, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k) {
        enqueue_gemm(-1, actor_id_, A, B, C, m, n, k, static_cast<T>(1.0), static_cast<T>(0.0), true);
      },
      // Mem ptr return overloads for already-existing Device buffers (explicit alpha, beta)
      [this](return_mem_ptr_atom, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k, T alpha, T beta) {
        enqueue_gemm(-1, actor_id_, A, B, C, m, n, k, alpha, beta, true);
      },
      // Mem ptr return overloads for already-existing Device buffers with routing (beta = 0.0)
      [this](return_mem_ptr_atom, int device_num, int stream_id, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k) {
        enqueue_gemm(device_num, stream_id, A, B, C, m, n, k, static_cast<T>(1.0), static_cast<T>(0.0), true);
      },
      // Mem ptr return overloads for already-existing Device buffers with routing (explicit alpha, beta)
      [this](return_mem_ptr_atom, int device_num, int stream_id, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k, T alpha, T beta) {
        enqueue_gemm(device_num, stream_id, A, B, C, m, n, k, alpha, beta, true);
      },
    };
  }

private:
  // Overload for Host-wrapped buffers
  template <typename OutType>
  void enqueue_gemm(int device_num, int stream_id,
                    in<T> A_arg, in<T> B_arg, OutType C_arg,
                    int m, int n, int k, T alpha, T beta, bool return_ptrs) {
    command_runner<in<T>, in<T>, OutType> runner;
    auto results = runner.transfer_memory(device_num, stream_id, A_arg, B_arg, C_arg);
    execute_and_reply(device_num, stream_id, std::get<0>(results),
                      std::get<1>(results), std::get<2>(results),
                      m, n, k, alpha, beta, return_ptrs);
  }

  // Overload for already-existing Device buffers
  void enqueue_gemm(int device_num, int stream_id,
                    mem_ptr<T> A_ptr, mem_ptr<T> B_ptr, mem_ptr<T> C_ptr,
                    int m, int n, int k, T alpha, T beta, bool return_ptrs) {
    command_runner<mem_ptr<T>, mem_ptr<T>, mem_ptr<T>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, A_ptr, B_ptr, C_ptr);
    execute_and_reply(device_num, stream_id, std::get<0>(results),
                      std::get<1>(results), std::get<2>(results),
                      m, n, k, alpha, beta, return_ptrs);
  }

  void execute_and_reply(int device_num, int stream_id,
                         mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C,
                         int m, int n, int k, T alpha, T beta, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev;
    if (device_num == -1)
      dev = plat->schedule(stream_id);
    else
      dev = plat->schedule(stream_id, device_num);

    // Dispatch based on template type T
    if constexpr (std::is_same_v<T, float>) {
      dev->sgemm(stream_id, m, n, k, alpha, A, B, beta, C);
    } else if constexpr (std::is_same_v<T, double>) {
      dev->dgemm(stream_id, m, n, k, alpha, A, B, beta, C);
    } else {
      static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, 
                    "Unsupported type for GEMM actor. Only float and double are supported.");
    }

    handle_reply(device_num, stream_id, A, B, C, return_ptrs);
  }

  void handle_reply(int, int, mem_ptr<T> A_ptr, mem_ptr<T> B_ptr, mem_ptr<T> C_ptr, bool return_ptrs) {
    command_runner<mem_ptr<T>> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;

    auto r_id = reply_id_;

    if (return_ptrs) {
      // Send all pointers in a single message
      caf::anon_mail(r_id, A_ptr, B_ptr, C_ptr).send(sender);
    } else {
      // Send result data in a single message
      runner.copy_to_host_async(C_ptr, [sender, r_id](std::vector<T>&& data) {
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