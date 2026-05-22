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

/// GEMM Batched Actor for performing matrix-matrix multiplication on batches of matrices.
/// Uses Strided Batched cuBLAS calls.
template <typename T>
class batched_gemm_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<batched_gemm_actor<T>>(reply_id);
  }

  batched_gemm_actor(caf::actor_config& cfg, int reply_id = 0)
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~batched_gemm_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      // Standard host buffer based calls (alpha=1.0, beta=0.0)
      [this](in<T> A, in<T> B, out<T> C, int m, int n, int k, int batchCount) {
        enqueue_gemm_batched(-1, actor_id_, A, B, C, m, n, k, batchCount, static_cast<T>(1.0), static_cast<T>(0.0), false);
      },
      // mem_ptr based calls (Zero-copy Pipelines)
      [this](mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k, int batchCount) {
        enqueue_gemm_batched(-1, actor_id_, A, B, C, m, n, k, batchCount, static_cast<T>(1.0), static_cast<T>(0.0), false);
      },
      // Return mem_ptr atom overloads
      [this](return_mem_ptr_atom, in<T> A, in<T> B, out<T> C, int m, int n, int k, int batchCount) {
        enqueue_gemm_batched(-1, actor_id_, A, B, C, m, n, k, batchCount, static_cast<T>(1.0), static_cast<T>(0.0), true);
      },
      [this](return_mem_ptr_atom, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k, int batchCount) {
        enqueue_gemm_batched(-1, actor_id_, A, B, C, m, n, k, batchCount, static_cast<T>(1.0), static_cast<T>(0.0), true);
      },
      // Routing control (Explicit device/stream)
      [this](int device_num, int stream_id, in<T> A, in<T> B, out<T> C, int m, int n, int k, int batchCount) {
        enqueue_gemm_batched(device_num, stream_id, A, B, C, m, n, k, batchCount, static_cast<T>(1.0), static_cast<T>(0.0), false);
      },
      [this](int device_num, int stream_id, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k, int batchCount) {
        enqueue_gemm_batched(device_num, stream_id, A, B, C, m, n, k, batchCount, static_cast<T>(1.0), static_cast<T>(0.0), false);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, in<T> A, in<T> B, out<T> C, int m, int n, int k, int batchCount) {
        enqueue_gemm_batched(device_num, stream_id, A, B, C, m, n, k, batchCount, static_cast<T>(1.0), static_cast<T>(0.0), true);
      },
      [this](return_mem_ptr_atom, int device_num, int stream_id, mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C, int m, int n, int k, int batchCount) {
        enqueue_gemm_batched(device_num, stream_id, A, B, C, m, n, k, batchCount, static_cast<T>(1.0), static_cast<T>(0.0), true);
      },
      // Full Parameter GEMM (custom alpha/beta)
      [this](in<T> A, in<T> B, out<T> C, int m, int n, int k, int batchCount, T alpha, T beta) {
        enqueue_gemm_batched(-1, actor_id_, A, B, C, m, n, k, batchCount, alpha, beta, false);
      },
      [this](return_mem_ptr_atom, in<T> A, in<T> B, out<T> C, int m, int n, int k, int batchCount, T alpha, T beta) {
        enqueue_gemm_batched(-1, actor_id_, A, B, C, m, n, k, batchCount, alpha, beta, true);
      }
    };
  }

private:
  // Logic for host-wrapped buffers
  template <typename OutType>
  void enqueue_gemm_batched(int device_num, int stream_id,
                            in<T> A_arg, in<T> B_arg, OutType C_arg,
                            int m, int n, int k, int batchCount, T alpha, T beta, bool return_ptrs) {
    command_runner<in<T>, in<T>, OutType> runner;
    auto results = runner.transfer_memory(device_num, stream_id, A_arg, B_arg, C_arg);
    execute_and_reply(device_num, stream_id, std::get<0>(results),
                      std::get<1>(results), std::get<2>(results),
                      m, n, k, batchCount, alpha, beta, return_ptrs);
  }

  // Logic for existing Device buffers
  void enqueue_gemm_batched(int device_num, int stream_id,
                            mem_ptr<T> A_ptr, mem_ptr<T> B_ptr, mem_ptr<T> C_ptr,
                            int m, int n, int k, int batchCount, T alpha, T beta, bool return_ptrs) {
    command_runner<mem_ptr<T>, mem_ptr<T>, mem_ptr<T>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, A_ptr, B_ptr, C_ptr);
    execute_and_reply(device_num, stream_id, std::get<0>(results),
                      std::get<1>(results), std::get<2>(results),
                      m, n, k, batchCount, alpha, beta, return_ptrs);
  }

  void execute_and_reply(int device_num, int stream_id,
                         mem_ptr<T> A, mem_ptr<T> B, mem_ptr<T> C,
                         int m, int n, int k, int batchCount, T alpha, T beta, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev = (device_num == -1) ? plat->schedule(stream_id) : plat->schedule(stream_id, device_num);

    // Default packed strides
    long long int strideA = static_cast<long long int>(m) * k;
    long long int strideB = static_cast<long long int>(k) * n;
    long long int strideC = static_cast<long long int>(m) * n;

    if constexpr (std::is_same_v<T, float>) {
      dev->sgemm_strided_batched(stream_id, m, n, k, alpha, A, strideA, B, strideB, beta, C, strideC, batchCount);
    } else if constexpr (std::is_same_v<T, double>) {
      dev->dgemm_strided_batched(stream_id, m, n, k, alpha, A, strideA, B, strideB, beta, C, strideC, batchCount);
    } else {
      static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, 
                    "Unsupported type for GEMM batched actor. Only float and double are supported.");
    }

    handle_reply(A, B, C, return_ptrs);
  }

  void handle_reply(mem_ptr<T> A_ptr, mem_ptr<T> B_ptr, mem_ptr<T> C_ptr, bool return_ptrs) {
    command_runner<mem_ptr<T>> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;

    auto r_id = reply_id_;

    if (return_ptrs) {
      // Requirement: Always 1 message. 
      // We bundle all pointers into a single message.
      caf::anon_mail(r_id, A_ptr, B_ptr, C_ptr).send(sender);
    } else {
      // Requirement: Always 1 message. 
      // We only copy back the result matrix (index 2) and send it.
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
