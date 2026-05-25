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

/// SPMM Actor for single-precision sparse matrix-matrix multiplication.
/// Message Signature: (csr_atom/csc_atom/coo_atom, in<int> row_ptr, in<int> col_ind, in<float> values, in<float> B, out<float> C, int m, int n, int k, int nnz, [float alpha, float beta])
class spmm_actor : public event_based_actor {
public:
  static caf::actor spawn(caf::actor_system& sys, int reply_id = 0) {
    return sys.spawn<spmm_actor>(reply_id);
  }

  spmm_actor(caf::actor_config& cfg, int reply_id = 0) 
    : event_based_actor(cfg), reply_id_(reply_id) {
    actor_id_ = static_cast<int>(this->id());
  }

  ~spmm_actor() override {
    command_runner<> runner;
    runner.release_stream_for_actor(actor_id_);
  }

  caf::behavior make_behavior() override {
    return {
      // CSR Overloads
      [this](csr_atom, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csr(-1, actor_id_, rp, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](csr_atom, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csr(-1, actor_id_, rp, ci, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](csr_atom, int dev, int sid, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csr(dev, sid, rp, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](csr_atom, int dev, int sid, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csr(dev, sid, rp, ci, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](csr_atom, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csr(-1, actor_id_, rp, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](csr_atom, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csr(-1, actor_id_, rp, ci, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](csr_atom, int dev, int sid, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csr(dev, sid, rp, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](csr_atom, int dev, int sid, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csr(dev, sid, rp, ci, v, b, c, m, n, k, nnz, alpha, beta, false);
      },

      // CSC Overloads
      [this](csc_atom, in<int> cp, in<int> ri, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csc(-1, actor_id_, cp, ri, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](csc_atom, in<int> cp, in<int> ri, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csc(-1, actor_id_, cp, ri, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](csc_atom, int dev, int sid, in<int> cp, in<int> ri, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csc(dev, sid, cp, ri, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](csc_atom, int dev, int sid, in<int> cp, in<int> ri, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csc(dev, sid, cp, ri, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](csc_atom, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csc(-1, actor_id_, cp, ri, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](csc_atom, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csc(-1, actor_id_, cp, ri, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](csc_atom, int dev, int sid, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csc(dev, sid, cp, ri, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](csc_atom, int dev, int sid, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csc(dev, sid, cp, ri, v, b, c, m, n, k, nnz, alpha, beta, false);
      },

      // COO Overloads
      [this](coo_atom, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_coo(-1, actor_id_, ri, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](coo_atom, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_coo(-1, actor_id_, ri, ci, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](coo_atom, int dev, int sid, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_coo(dev, sid, ri, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](coo_atom, int dev, int sid, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_coo(dev, sid, ri, ci, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](coo_atom, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_coo(-1, actor_id_, ri, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](coo_atom, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_coo(-1, actor_id_, ri, ci, v, b, c, m, n, k, nnz, alpha, beta, false);
      },
      [this](coo_atom, int dev, int sid, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_coo(dev, sid, ri, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, false);
      },
      [this](coo_atom, int dev, int sid, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_coo(dev, sid, ri, ci, v, b, c, m, n, k, nnz, alpha, beta, false);
      },

      // return_mem_ptr_atom variants (CSR example)
      [this](return_mem_ptr_atom, csr_atom, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csr(-1, actor_id_, rp, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csr_atom, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csr(-1, actor_id_, rp, ci, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csr_atom, int dev, int sid, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csr(dev, sid, rp, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csr_atom, int dev, int sid, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csr(dev, sid, rp, ci, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csr_atom, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csr(-1, actor_id_, rp, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csr_atom, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csr(-1, actor_id_, rp, ci, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csr_atom, int dev, int sid, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csr(dev, sid, rp, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csr_atom, int dev, int sid, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csr(dev, sid, rp, ci, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      // return_mem_ptr_atom variants (CSC)
      [this](return_mem_ptr_atom, csc_atom, in<int> cp, in<int> ri, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csc(-1, actor_id_, cp, ri, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csc_atom, int dev, int sid, in<int> cp, in<int> ri, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csc(dev, sid, cp, ri, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csc_atom, int dev, int sid, in<int> cp, in<int> ri, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csc(dev, sid, cp, ri, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csc_atom, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csc(-1, actor_id_, cp, ri, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csc_atom, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csc(-1, actor_id_, cp, ri, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, csc_atom, int dev, int sid, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_csc(dev, sid, cp, ri, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, csc_atom, int dev, int sid, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_csc(dev, sid, cp, ri, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, coo_atom, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_coo(-1, actor_id_, ri, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, coo_atom, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_coo(-1, actor_id_, ri, ci, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, coo_atom, int dev, int sid, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_coo(dev, sid, ri, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, coo_atom, int dev, int sid, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_coo(dev, sid, ri, ci, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, coo_atom, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_coo(-1, actor_id_, ri, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, coo_atom, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_coo(-1, actor_id_, ri, ci, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
      [this](return_mem_ptr_atom, coo_atom, int dev, int sid, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz) {
        enqueue_spmm_coo(dev, sid, ri, ci, v, b, c, m, n, k, nnz, 1.0f, 0.0f, true);
      },
      [this](return_mem_ptr_atom, coo_atom, int dev, int sid, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c, int m, int n, int k, int nnz, float alpha, float beta) {
        enqueue_spmm_coo(dev, sid, ri, ci, v, b, c, m, n, k, nnz, alpha, beta, true);
      },
    };
  }

private:
  // CSR Enqueue logic
  void enqueue_spmm_csr(int dev, int sid, in<int> rp, in<int> ci, in<float> v, in<float> b, out<float> c,
                        int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
    auto res = runner.transfer_memory(dev, sid, rp, ci, v, b, c);
    execute_and_reply_csr(dev, sid, std::get<0>(res), std::get<1>(res), std::get<2>(res), 
                          std::get<3>(res), std::get<4>(res), m, n, k, nnz, alpha, beta, ret_ptr);
  }

  void enqueue_spmm_csr(int dev, int sid, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c,
                        int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    command_runner<mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, mem_ptr<float>, mem_ptr<float>> runner;
    auto res = runner.transfer_memory(dev, sid, rp, ci, v, b, c);
    execute_and_reply_csr(dev, sid, std::get<0>(res), std::get<1>(res), std::get<2>(res), 
                          std::get<3>(res), std::get<4>(res), m, n, k, nnz, alpha, beta, ret_ptr);
  }

  void execute_and_reply_csr(int dev_n, int sid, mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c,
                             int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    auto plat = platform::create();
    device_ptr dev = (dev_n == -1) ? plat->schedule(sid) : plat->schedule(sid, dev_n);
    dev->spmm_csr(sid, m, n, k, nnz, alpha, rp, ci, v, b, beta, c);
    handle_reply(rp, ci, v, b, c, ret_ptr);
  }

  // CSC Enqueue logic
  void enqueue_spmm_csc(int dev, int sid, in<int> cp, in<int> ri, in<float> v, in<float> b, out<float> c,
                        int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
    auto res = runner.transfer_memory(dev, sid, cp, ri, v, b, c);
    execute_and_reply_csc(dev, sid, std::get<0>(res), std::get<1>(res), std::get<2>(res), 
                          std::get<3>(res), std::get<4>(res), m, n, k, nnz, alpha, beta, ret_ptr);
  }

  void enqueue_spmm_csc(int dev, int sid, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c,
                        int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    command_runner<mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, mem_ptr<float>, mem_ptr<float>> runner;
    auto res = runner.transfer_memory(dev, sid, cp, ri, v, b, c);
    execute_and_reply_csc(dev, sid, std::get<0>(res), std::get<1>(res), std::get<2>(res), 
                          std::get<3>(res), std::get<4>(res), m, n, k, nnz, alpha, beta, ret_ptr);
  }

  void execute_and_reply_csc(int dev_n, int sid, mem_ptr<int> cp, mem_ptr<int> ri, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c,
                             int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    auto plat = platform::create();
    device_ptr dev = (dev_n == -1) ? plat->schedule(sid) : plat->schedule(sid, dev_n);
    dev->spmm_csc(sid, m, n, k, nnz, alpha, cp, ri, v, b, beta, c);
    handle_reply(cp, ri, v, b, c, ret_ptr);
  }

  // COO Enqueue logic
  void enqueue_spmm_coo(int dev, int sid, in<int> ri, in<int> ci, in<float> v, in<float> b, out<float> c,
                        int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
    auto res = runner.transfer_memory(dev, sid, ri, ci, v, b, c);
    execute_and_reply_coo(dev, sid, std::get<0>(res), std::get<1>(res), std::get<2>(res), 
                          std::get<3>(res), std::get<4>(res), m, n, k, nnz, alpha, beta, ret_ptr);
  }

  void enqueue_spmm_coo(int dev, int sid, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c,
                        int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    command_runner<mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, mem_ptr<float>, mem_ptr<float>> runner;
    auto res = runner.transfer_memory(dev, sid, ri, ci, v, b, c);
    execute_and_reply_coo(dev, sid, std::get<0>(res), std::get<1>(res), std::get<2>(res), 
                          std::get<3>(res), std::get<4>(res), m, n, k, nnz, alpha, beta, ret_ptr);
  }

  void execute_and_reply_coo(int dev_n, int sid, mem_ptr<int> ri, mem_ptr<int> ci, mem_ptr<float> v, mem_ptr<float> b, mem_ptr<float> c,
                             int m, int n, int k, int nnz, float alpha, float beta, bool ret_ptr) {
    auto plat = platform::create();
    device_ptr dev = (dev_n == -1) ? plat->schedule(sid) : plat->schedule(sid, dev_n);
    dev->spmm_coo(sid, m, n, k, nnz, alpha, ri, ci, v, b, beta, c);
    handle_reply(ri, ci, v, b, c, ret_ptr);
  }

  template <typename P1, typename P2, typename P3, typename P4, typename P5>
  void handle_reply(P1 p1, P2 p2, P3 p3, P4 p4, P5 p5, bool return_ptrs) {
    command_runner<mem_ptr<float>> runner;
    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;
    auto r_id = reply_id_;
    if (return_ptrs) {
      caf::anon_mail(r_id, p1, p2, p3, p4, p5).send(sender);
    } else {
      runner.copy_to_host_async(p5, [sender, r_id](std::vector<float>&& data) {
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