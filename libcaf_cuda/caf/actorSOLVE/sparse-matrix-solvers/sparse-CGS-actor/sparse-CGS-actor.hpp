#pragma once

#include <caf/all.hpp>
#include <deque>
#include <vector>
#include "caf/cuda/all.hpp"
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"
#include "caf/actorSPARSE/spmv-actor/spmv-actor.hpp"
#include "caf/actorBLAS/axpy-actor/axpy-actor.hpp"
#include "caf/actorBLAS/copy-actor/copy-actor.hpp"

namespace caf::cuda {

enum class matrix_format {
  csr,
  csc,
  coo
};

enum class sparse_cg_step {
  idle,
  init_r,
  init_p,
  init_rho,
  main_spmv_w,
  main_dot_pw,
  main_axpy_x,
  main_axpy_r,
  main_dot_rr,
  update_p_copy_r,
  update_p_axpy_p,
  update_p_final_copy
};

// Reply IDs used to distinguish which actor type is replying
constexpr int id_dot = 100;
constexpr int id_spmv = 200;
constexpr int id_axpy = 300;
constexpr int id_copy = 400;

struct sparse_cg_state {
  // Host Data (stored until start)
  in<int> h_row_ptr, h_col_ind;
  in<float> h_values, h_b;
  in_out<float> h_x;

  // Device Problem data
  mem_ptr<int> A_row_ptr, A_col_ind;
  mem_ptr<float> A_values, b, x;
  
  matrix_format format;
  int n, nnz;
  float tol;
  int max_iter;
  int device_num;
  int stream_id;

  // Workspace vectors
  mem_ptr<float> r, p, w, y_tmp;
  
  // Scalars
  float rho = 0.0f;
  float old_rho = 0.0f;
  float alpha = 0.0f;
  float beta = 0.0f;
  int iterations = 0;
  sparse_cg_step step = sparse_cg_step::idle;

  // BLAS/SPARSE Actors
  caf::actor dot_actor, spmv_actor, axpy_actor, copy_actor;
  caf::actor supervisor;
};

class sparse_cg_actor : public stateful_actor<sparse_cg_state> {
public:
  sparse_cg_actor(actor_config& cfg, in<int> rp, in<int> ci, 
                  in<float> val, in<float> b, in_out<float> x,
                  matrix_format fmt, int n, int nnz, float tol, int max_iter, int device_num, int stream_id,
                  caf::actor supervisor = nullptr)
    : stateful_actor<sparse_cg_state>(cfg) {
    state().h_row_ptr = std::move(rp);
    state().h_col_ind = std::move(ci);
    state().h_values = std::move(val);
    state().h_b = std::move(b);
    state().h_x = std::move(x);
    state().format = fmt;
    state().n = n; state().nnz = nnz; 
    state().tol = tol; state().max_iter = max_iter;
    state().device_num = device_num; state().stream_id = stream_id;
    state().supervisor = supervisor;

    // Spawn helpers
    state().dot_actor = this->system().spawn<caf::cuda::dot_actor>(id_dot);
    state().spmv_actor = this->system().spawn<caf::cuda::spmv_actor>(id_spmv);
    state().axpy_actor = this->system().spawn<caf::cuda::axpy_actor>(id_axpy);
    state().copy_actor = this->system().spawn<caf::cuda::copy_actor>(id_copy);
  }

  behavior make_behavior() override {
    return {
      [this](start_atom) {
        if (!state().supervisor)
          state().supervisor = actor_cast<caf::actor>(this->current_sender());
        start_setup();
      },
      [this](int rid, float val) {
        if (rid == id_dot) handle_dot_result(val);
      },
      // SPMV result (CSR Signature: rid, rp, ci, val, x, y)
      [this](int rid, mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, mem_ptr<float>, mem_ptr<float>) {
        if (rid == id_spmv) handle_spmv_result();
      },
      [this](int rid, mem_ptr<float> x, mem_ptr<float> y) {
        if (rid == id_copy) handle_copy_result();
        else if (rid == id_axpy) handle_axpy_result();
      }
    };
  }

private:
  void start_setup() {
    auto& s = state();
    command_runner<> runner;
    
    // Transfer problem data to device
    auto res = runner.transfer_memory(s.device_num, s.stream_id, 
                                      s.h_row_ptr,
                                      s.h_col_ind,
                                      s.h_values,
                                      s.h_b,
                                      s.h_x);
    
    s.A_row_ptr = std::get<0>(res);
    s.A_col_ind = std::get<1>(res);
    s.A_values  = std::get<2>(res);
    s.b         = std::get<3>(res);
    s.x         = std::get<4>(res);

    // Allocate workspace
    command_runner<out<float>> work_runner;
    s.r = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.p = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.w = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));

    // Start algorithm: r = b
    s.step = sparse_cg_step::init_r;
    this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.b, s.r, s.n).send(s.copy_actor);
  }

  void iterate() {
    auto& s = state();
    if (s.iterations >= s.max_iter || s.rho < (s.tol * s.tol)) {
      if (s.supervisor) {
        command_runner<float> cr;
        cr.copy_to_host_async(s.x, [this, target = s.supervisor](std::vector<float>&& data) {
          this->mail(std::move(data)).send(target);
        });
      }
      s.step = sparse_cg_step::idle;
      return;
    }
    s.step = sparse_cg_step::main_spmv_w;
    send_spmv(s.p, s.w);
  }

  void send_spmv(mem_ptr<float> input_v, mem_ptr<float> output_v) {
    auto& s = state();
    auto send = [&](auto format_atom) {
      this->mail(return_mem_ptr_atom_v, format_atom, s.device_num, s.stream_id, 
                 s.A_row_ptr, s.A_col_ind, s.A_values, input_v, output_v, 
                 s.n, s.n, s.nnz).send(s.spmv_actor);
    };

    switch (s.format) {
      case matrix_format::csr: send(csr_atom_v); break;
      case matrix_format::csc: send(csc_atom_v); break;
      case matrix_format::coo: send(coo_atom_v); break;
      default: break;
    }
  }

  void handle_dot_result(float val) {
    auto& s = state();
    switch (s.step) {
      case sparse_cg_step::init_rho:
        s.rho = val;
        iterate();
        break;
      case sparse_cg_step::main_dot_pw:
        s.alpha = s.rho / val;
        s.step = sparse_cg_step::main_axpy_x;
        this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.p, s.x, s.n, s.alpha).send(s.axpy_actor);
        break;
      case sparse_cg_step::main_dot_rr:
        s.old_rho = s.rho;
        s.rho = val;
        s.iterations++;
        s.beta = s.rho / s.old_rho;
        s.step = sparse_cg_step::update_p_copy_r;
        this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.r, s.w, s.n).send(s.copy_actor);
        break;
      default: break;
    }
  }

  void handle_spmv_result() {
    auto& s = state();
    if (s.step == sparse_cg_step::main_spmv_w) {
      s.step = sparse_cg_step::main_dot_pw;
      this->mail(s.device_num, s.stream_id, s.p, s.w, s.y_tmp, s.n).send(s.dot_actor);
    }
  }

  void handle_axpy_result() {
    auto& s = state();
    if (s.step == sparse_cg_step::main_axpy_x) {
      s.step = sparse_cg_step::main_axpy_r;
      this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.w, s.r, s.n, -s.alpha).send(s.axpy_actor);
    } else if (s.step == sparse_cg_step::main_axpy_r) {
      s.step = sparse_cg_step::main_dot_rr;
      this->mail(s.device_num, s.stream_id, s.r, s.r, s.y_tmp, s.n).send(s.dot_actor);
    } else if (s.step == sparse_cg_step::update_p_axpy_p) {
      s.step = sparse_cg_step::update_p_final_copy;
      this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.w, s.p, s.n).send(s.copy_actor);
    }
    else if (s.step == sparse_cg_step::idle) return;
  }

  void handle_copy_result() {
    auto& s = state();
    switch (s.step) {
      case sparse_cg_step::init_r:
        s.step = sparse_cg_step::init_p;
        this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.r, s.p, s.n).send(s.copy_actor);
        break;
      case sparse_cg_step::init_p:
        s.step = sparse_cg_step::init_rho;
        this->mail(s.device_num, s.stream_id, s.r, s.r, s.y_tmp, s.n).send(s.dot_actor);
        break;
      case sparse_cg_step::update_p_copy_r:
        s.step = sparse_cg_step::update_p_axpy_p;
        this->mail(return_mem_ptr_atom_v, s.device_num, s.stream_id, s.p, s.w, s.n, s.beta).send(s.axpy_actor);
        break;
      case sparse_cg_step::update_p_final_copy:
        iterate();
        break;
      default: break;
    }
  }
};

} // namespace caf::cuda
