#pragma once

#include <caf/all.hpp>
#include <vector>
#include "caf/cuda/all.hpp"
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"
#include "caf/actorSPARSE/spmv-actor/spmv-actor.hpp"
#include "caf/actorBLAS/axpy-actor/axpy-actor.hpp"
#include "caf/actorBLAS/copy-actor/copy-actor.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-CGS-actor/sparse-CGS-actor.hpp"

namespace caf::cuda {

struct sparse_bicgstab_state {
  // Host Data
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
  caf::actor supervisor;

  // Workspace vectors
  mem_ptr<float> r, r_hat, p, v, s_vec, t_vec, y_tmp;
  
  // Scalars
  float rho = 1.0f;
  float alpha = 1.0f;
  float omega = 1.0f;
  float beta = 0.0f;
  int iterations = 0;
};

class sparse_bicgstab_actor : public stateful_actor<sparse_bicgstab_state> {
public:
  sparse_bicgstab_actor(actor_config& cfg, in<int> rp, in<int> ci,
                        in<float> val, in<float> b, in_out<float> x,
                        matrix_format fmt, int n, int nnz, float tol, int max_iter, int device_num, int stream_id,
                        caf::actor supervisor = nullptr)
    : stateful_actor<sparse_bicgstab_state>(cfg) {
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
  }

  behavior make_behavior() override {
    return {
      [this](start_atom) {
        auto& s = state();
        if (!s.supervisor)
          s.supervisor = actor_cast<caf::actor>(this->current_sender());
        start_solve();
      },
    };
  }

private:
  void start_solve() {
    auto& s = state();
    command_runner<> runner;

    // Transfer problem data to device
    auto res = runner.transfer_memory(s.device_num, s.stream_id,
                                      s.h_row_ptr, s.h_col_ind,
                                      s.h_values, s.h_b, s.h_x);
   
    s.A_row_ptr = std::get<0>(res);
    s.A_col_ind = std::get<1>(res);
    s.A_values  = std::get<2>(res);
    s.b         = std::get<3>(res);
    s.x         = std::get<4>(res);

    // Allocate workspace
    command_runner<out<float>> work_runner;
    s.r     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.r_hat = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.p     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.v     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.s_vec = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.t_vec = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));

    auto dev = platform::create()->schedule(s.stream_id, s.device_num);

    // 1. Initial Residual: r = b - Ax
    execute_spmv(s.x, s.v); 
    dev->scopy(s.stream_id, s.n, s.b, s.r);
    dev->saxpy(s.stream_id, s.n, -1.0f, s.v, s.r);
    
    // 2. Choose r_hat = r
    dev->scopy(s.stream_id, s.n, s.r, s.r_hat);

    s.iterations = 0;
    float residual_norm_sq = 0.0f;
    dev->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
    residual_norm_sq = s.y_tmp->copy_to_host()[0];

    while (residual_norm_sq > (s.tol * s.tol) && s.iterations < s.max_iter) {
      // rho_i = <r_hat, r>
      dev->sdot(s.stream_id, s.n, s.r_hat, s.r, s.y_tmp);
      float rho_new = s.y_tmp->copy_to_host()[0];

      if (s.iterations == 0) {
        dev->scopy(s.stream_id, s.n, s.r, s.p);
      } else {
        s.beta = (rho_new / s.rho) * (s.alpha / s.omega);
        // p = r + beta * (p - omega * v)
        dev->saxpy(s.stream_id, s.n, -s.omega, s.v, s.p); // p = p - omega*v
        dev->scopy(s.stream_id, s.n, s.r, s.s_vec);      // use s_vec as temporary
        dev->saxpy(s.stream_id, s.n, s.beta, s.p, s.s_vec); // s_vec = r + beta*p
        dev->scopy(s.stream_id, s.n, s.s_vec, s.p);
      }

      s.rho = rho_new;

      // v = Ap
      execute_spmv(s.p, s.v);

      // alpha = rho / <r_hat, v>
      dev->sdot(s.stream_id, s.n, s.r_hat, s.v, s.y_tmp);
      s.alpha = s.rho / s.y_tmp->copy_to_host()[0];

      // s = r - alpha * v
      dev->scopy(s.stream_id, s.n, s.r, s.s_vec);
      dev->saxpy(s.stream_id, s.n, -s.alpha, s.v, s.s_vec);

      // t = As
      execute_spmv(s.s_vec, s.t_vec);

      // omega = <t, s> / <t, t>
      dev->sdot(s.stream_id, s.n, s.t_vec, s.s_vec, s.y_tmp);
      float dot_ts = s.y_tmp->copy_to_host()[0];
      dev->sdot(s.stream_id, s.n, s.t_vec, s.t_vec, s.y_tmp);
      float dot_tt = s.y_tmp->copy_to_host()[0];
      s.omega = dot_ts / dot_tt;

      // x = x + alpha*p + omega*s
      dev->saxpy(s.stream_id, s.n, s.alpha, s.p, s.x);
      dev->saxpy(s.stream_id, s.n, s.omega, s.s_vec, s.x);

      // r = s - omega*t
      dev->scopy(s.stream_id, s.n, s.s_vec, s.r);
      dev->saxpy(s.stream_id, s.n, -s.omega, s.t_vec, s.r);

      dev->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
      residual_norm_sq = s.y_tmp->copy_to_host()[0];
      s.iterations++;
    }

    auto solution = s.x->copy_to_host();
    if (s.supervisor)
      this->mail(std::move(solution)).send(s.supervisor);
  }

  void execute_spmv(mem_ptr<float> input_v, mem_ptr<float> output_v) {
    auto& s = state();
    auto dev = platform::create()->schedule(s.stream_id, s.device_num);
    switch (s.format) {
      case matrix_format::csr: dev->spmv_csr(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v); break;
      case matrix_format::csc: dev->spmv_csc(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v); break;
      case matrix_format::coo: dev->spmv_coo(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v); break;
      default: break;
    }
  }
};

} // namespace caf::cuda