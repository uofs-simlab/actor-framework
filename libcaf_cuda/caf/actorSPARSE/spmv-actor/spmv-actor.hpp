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

/// SPMV Actor for single-precision sparse matrix-vector multiplication.
/// Message Signature: (csr_atom, in<int> row_ptr, in<int> col_ind, in<float> values, in<float> x, out<float> y, int m, int n, int nnz, [float alpha, float beta])
class spmv_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<spmv_actor>(reply_id);
  }

  spmv_actor(caf::actor_config& cfg, int reply_id = 0) 
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~spmv_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      // CSR Host-based overloads
      [this](csr_atom, in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_csr(-1, actor_id_, row_ptr, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](csr_atom, in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csr(-1, actor_id_, row_ptr, col_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // CSR Routing overloads
      [this](csr_atom, int device_num, int stream_id, in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_csr(device_num, stream_id, row_ptr, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](csr_atom, int device_num, int stream_id, in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csr(device_num, stream_id, row_ptr, col_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // CSR mem_ptr overloads
      [this](csr_atom, mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_csr(-1, actor_id_, row_ptr, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](csr_atom, mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csr(-1, actor_id_, row_ptr, col_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      [this](csr_atom, int device_num, int stream_id, mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_csr(device_num, stream_id, row_ptr, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](csr_atom, int device_num, int stream_id, mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csr(device_num, stream_id, row_ptr, col_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // CSC Host-based overloads
      [this](csc_atom, in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_csc(-1, actor_id_, col_ptr, row_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](csc_atom, in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csc(-1, actor_id_, col_ptr, row_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // CSC Routing overloads
      [this](csc_atom, int device_num, int stream_id, in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_csc(device_num, stream_id, col_ptr, row_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](csc_atom, int device_num, int stream_id, in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csc(device_num, stream_id, col_ptr, row_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // CSC mem_ptr overloads
      [this](csc_atom, mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_csc(-1, actor_id_, col_ptr, row_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](csc_atom, mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csc(-1, actor_id_, col_ptr, row_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      [this](csc_atom, int device_num, int stream_id, mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_csc(device_num, stream_id, col_ptr, row_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](csc_atom, int device_num, int stream_id, mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csc(device_num, stream_id, col_ptr, row_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // COO Host-based overloads
      [this](coo_atom, in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_coo(-1, actor_id_, row_ind, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](coo_atom, in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_coo(-1, actor_id_, row_ind, col_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // COO Routing overloads
      [this](coo_atom, int device_num, int stream_id, in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_coo(device_num, stream_id, row_ind, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](coo_atom, int device_num, int stream_id, in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_coo(device_num, stream_id, row_ind, col_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // COO mem_ptr overloads
      [this](coo_atom, mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_coo(-1, actor_id_, row_ind, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](coo_atom, mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_coo(-1, actor_id_, row_ind, col_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      [this](coo_atom, int device_num, int stream_id, mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_coo(device_num, stream_id, row_ind, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, false);
      },
      [this](coo_atom, int device_num, int stream_id, mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_coo(device_num, stream_id, row_ind, col_ind, val, x, y, m, n, nnz, alpha, beta, false);
      },
      // return_mem_ptr_atom variants
      [this](return_mem_ptr_atom, csr_atom, in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_csr(-1, actor_id_, row_ptr, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csr_atom, in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csr(-1, actor_id_, row_ptr, col_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csr_atom, int device_num, int stream_id, in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_csr(device_num, stream_id, row_ptr, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csr_atom, int device_num, int stream_id, in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csr(device_num, stream_id, row_ptr, col_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csr_atom, mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_csr(-1, actor_id_, row_ptr, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csr_atom, mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csr(-1, actor_id_, row_ptr, col_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csr_atom, int device_num, int stream_id, mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_csr(device_num, stream_id, row_ptr, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csr_atom, int device_num, int stream_id, mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csr(device_num, stream_id, row_ptr, col_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      // CSC return_mem_ptr_atom variants
      [this](return_mem_ptr_atom, csc_atom, in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_csc(-1, actor_id_, col_ptr, row_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csc_atom, in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csc(-1, actor_id_, col_ptr, row_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csc_atom, int device_num, int stream_id, in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_csc(device_num, stream_id, col_ptr, row_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csc_atom, int device_num, int stream_id, in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csc(device_num, stream_id, col_ptr, row_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csc_atom, mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_csc(-1, actor_id_, col_ptr, row_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csc_atom, mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csc(-1, actor_id_, col_ptr, row_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csc_atom, int device_num, int stream_id, mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_csc(device_num, stream_id, col_ptr, row_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csc_atom, int device_num, int stream_id, mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_csc(device_num, stream_id, col_ptr, row_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      // COO return_mem_ptr_atom variants
      [this](return_mem_ptr_atom, coo_atom, in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_coo(-1, actor_id_, row_ind, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, coo_atom, in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_coo(-1, actor_id_, row_ind, col_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, coo_atom, int device_num, int stream_id, in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz) {
        enqueue_spmv_coo(device_num, stream_id, row_ind, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, coo_atom, int device_num, int stream_id, in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_coo(device_num, stream_id, row_ind, col_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, coo_atom, mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_coo(-1, actor_id_, row_ind, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, coo_atom, mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_coo(-1, actor_id_, row_ind, col_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, coo_atom, int device_num, int stream_id, mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz) {
        enqueue_spmv_coo(device_num, stream_id, row_ind, col_ind, val, x, y, m, n, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, coo_atom, int device_num, int stream_id, mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, int m, int n, int nnz, float alpha, float beta) {
        enqueue_spmv_coo(device_num, stream_id, row_ind, col_ind, val, x, y, m, n, nnz, alpha, beta, true);
      },
    };
  }

private:
  void enqueue_spmv_csr(int device_num, int stream_id, 
                        in<int> row_ptr, in<int> col_ind, in<float> val, in<float> x, out<float> y, 
                        int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, row_ptr, col_ind, val, x, y);
    execute_and_reply_csr(device_num, stream_id, std::get<0>(results), std::get<1>(results), 
                          std::get<2>(results), std::get<3>(results), std::get<4>(results), 
                          m, n, nnz, alpha, beta, return_ptrs);
  }

  void enqueue_spmv_csr(int device_num, int stream_id, 
                        mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, 
                        int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    command_runner<mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, row_ptr, col_ind, val, x, y);
    execute_and_reply_csr(device_num, stream_id, std::get<0>(results), std::get<1>(results), 
                          std::get<2>(results), std::get<3>(results), std::get<4>(results), 
                          m, n, nnz, alpha, beta, return_ptrs);
  }

  void execute_and_reply_csr(int device_num, int stream_id, 
                             mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, 
                             int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev = (device_num == -1) ? plat->schedule(stream_id) : plat->schedule(stream_id, device_num);
    dev->spmv_csr(stream_id, m, n, nnz, alpha, row_ptr, col_ind, val, x, beta, y);
    handle_reply(device_num, stream_id, row_ptr, col_ind, val, x, y, return_ptrs);
  }

  // CSC Logic
  void enqueue_spmv_csc(int device_num, int stream_id, 
                        in<int> col_ptr, in<int> row_ind, in<float> val, in<float> x, out<float> y, 
                        int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, col_ptr, row_ind, val, x, y);
    execute_and_reply_csc(device_num, stream_id, std::get<0>(results), std::get<1>(results), 
                          std::get<2>(results), std::get<3>(results), std::get<4>(results), 
                          m, n, nnz, alpha, beta, return_ptrs);
  }

  void enqueue_spmv_csc(int device_num, int stream_id, 
                        mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, 
                        int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    command_runner<mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, col_ptr, row_ind, val, x, y);
    execute_and_reply_csc(device_num, stream_id, std::get<0>(results), std::get<1>(results), 
                          std::get<2>(results), std::get<3>(results), std::get<4>(results), 
                          m, n, nnz, alpha, beta, return_ptrs);
  }

  void execute_and_reply_csc(int device_num, int stream_id, 
                             mem_ptr<int> col_ptr, mem_ptr<int> row_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, 
                             int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev = (device_num == -1) ? plat->schedule(stream_id) : plat->schedule(stream_id, device_num);
    dev->spmv_csc(stream_id, m, n, nnz, alpha, col_ptr, row_ind, val, x, beta, y);
    handle_reply(device_num, stream_id, col_ptr, row_ind, val, x, y, return_ptrs);
  }

  // COO Logic
  void enqueue_spmv_coo(int device_num, int stream_id, 
                        in<int> row_ind, in<int> col_ind, in<float> val, in<float> x, out<float> y, 
                        int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, row_ind, col_ind, val, x, y);
    execute_and_reply_coo(device_num, stream_id, std::get<0>(results), std::get<1>(results), 
                          std::get<2>(results), std::get<3>(results), std::get<4>(results), 
                          m, n, nnz, alpha, beta, return_ptrs);
  }

  void enqueue_spmv_coo(int device_num, int stream_id, 
                        mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, 
                        int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    command_runner<mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, mem_ptr<float>, mem_ptr<float>> runner;
    auto results = runner.transfer_memory(device_num, stream_id, row_ind, col_ind, val, x, y);
    execute_and_reply_coo(device_num, stream_id, std::get<0>(results), std::get<1>(results), 
                          std::get<2>(results), std::get<3>(results), std::get<4>(results), 
                          m, n, nnz, alpha, beta, return_ptrs);
  }

  void execute_and_reply_coo(int device_num, int stream_id, 
                             mem_ptr<int> row_ind, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, 
                             int m, int n, int nnz, float alpha, float beta, bool return_ptrs) {
    auto plat = platform::create();
    device_ptr dev = (device_num == -1) ? plat->schedule(stream_id) : plat->schedule(stream_id, device_num);
    dev->spmv_coo(stream_id, m, n, nnz, alpha, row_ind, col_ind, val, x, beta, y);
    handle_reply(device_num, stream_id, row_ind, col_ind, val, x, y, return_ptrs);
  }

  void handle_reply(int, int, 
                    mem_ptr<int> row_ptr, mem_ptr<int> col_ind, mem_ptr<float> val, mem_ptr<float> x, mem_ptr<float> y, 
                    bool return_ptrs) {
    command_runner<mem_ptr<float>> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;
    auto r_id = reply_id_;
    if (return_ptrs) {
      caf::anon_mail(r_id, row_ptr, col_ind, val, x, y).send(sender);
    } else {
      runner.copy_to_host_async(y, [sender, r_id](std::vector<float>&& data) {
        if (sender) {
          caf::anon_mail(r_id, 4, std::move(data)).send(sender);
        }
      });
    }
  }

  int actor_id_;
  int reply_id_;
};

} // namespace caf::cuda