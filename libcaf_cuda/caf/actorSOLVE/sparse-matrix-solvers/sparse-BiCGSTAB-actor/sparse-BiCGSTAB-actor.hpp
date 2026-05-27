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
  device_ptr d_ptr;
  caf::actor supervisor;

  // Workspace vectors
  mem_ptr<float> r, r_hat, p, v, s_vec, t_vec, y_tmp;
  mem_ptr<char> spmv_workspace;
  
  // Scalars needed across asynchronous steps
  float rho_val = 1.0f; // Renamed from rho to avoid conflict with step-specific rho_new
  float alpha_val = 1.0f;
  float omega_val = 1.0f;
  float beta_val = 0.0f;
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

    s.d_ptr = platform::create()->schedule(s.stream_id, s.device_num);

    // Allocate workspace
    command_runner<out<float>> work_runner;
    s.r     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.r_hat = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.p     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.v     = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.s_vec = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    s.t_vec = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
    // Bottleneck Fix: Allocate only 1 float for scalar results
    s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(1));

    // Allocate SPMV workspace to avoid reallocations in the loop
    size_t ws_size = 0;
    if (s.format == matrix_format::csr)
      ws_size = s.d_ptr->spmv_csr_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.v);
    else if (s.format == matrix_format::csc)
      ws_size = s.d_ptr->spmv_csc_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.v);
    else if (s.format == matrix_format::coo)
      ws_size = s.d_ptr->spmv_coo_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.v);

    if (ws_size > 0) {
      command_runner<out<char>> ws_runner;
      s.spmv_workspace = ws_runner.transfer_memory(s.device_num, s.stream_id, out<char>(static_cast<int>(ws_size)));
    }

    // 1. Initial Residual: r = b - Ax
    execute_spmv(s.x, s.v); 
    s.d_ptr->scopy(s.stream_id, s.n, s.b, s.r);
    s.d_ptr->saxpy(s.stream_id, s.n, -1.0f, s.v, s.r); // r = b - Ax
    
    // 2. Choose r_hat = r
    s.d_ptr->scopy(s.stream_id, s.n, s.r, s.r_hat);

    // 3. Initial norm calculation
    s.d_ptr->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
    float norm_sq = s.y_tmp->copy_to_host()[0];

    s.iterations = 0;
    s.rho_val = 1.0f;
    s.alpha_val = 1.0f;
    s.omega_val = 1.0f;

    // BiCGSTAB Loop
    while (norm_sq > (s.tol * s.tol) && s.iterations < s.max_iter) {
      s.iterations++;

      // rho_new = <r_hat, r>
      s.d_ptr->sdot(s.stream_id, s.n, s.r_hat, s.r, s.y_tmp);
      float rho_new = s.y_tmp->copy_to_host()[0];

      if (s.iterations == 1) {
        s.d_ptr->scopy(s.stream_id, s.n, s.r, s.p);
      } else {
        s.beta_val = (rho_new / s.rho_val) * (s.alpha_val / s.omega_val);
        // p = r + beta * (p - omega * v)
        s.d_ptr->saxpy(s.stream_id, s.n, -s.omega_val, s.v, s.p);
        s.d_ptr->scopy(s.stream_id, s.n, s.r, s.s_vec);
        s.d_ptr->saxpy(s.stream_id, s.n, s.beta_val, s.p, s.s_vec);
        s.d_ptr->scopy(s.stream_id, s.n, s.s_vec, s.p);
      }
      s.rho_val = rho_new;

      // v = Ap
      execute_spmv(s.p, s.v);

      // alpha = rho / <r_hat, v>
      s.d_ptr->sdot(s.stream_id, s.n, s.r_hat, s.v, s.y_tmp);
      float alpha_denom = s.y_tmp->copy_to_host()[0];
      s.alpha_val = s.rho_val / alpha_denom;

      // s = r - alpha * v
      s.d_ptr->scopy(s.stream_id, s.n, s.r, s.s_vec);
      s.d_ptr->saxpy(s.stream_id, s.n, -s.alpha_val, s.v, s.s_vec);

      // t = As
      execute_spmv(s.s_vec, s.t_vec);

      // omega = <t, s> / <t, t>
      s.d_ptr->sdot(s.stream_id, s.n, s.t_vec, s.s_vec, s.y_tmp);
      float omega_num = s.y_tmp->copy_to_host()[0];
      s.d_ptr->sdot(s.stream_id, s.n, s.t_vec, s.t_vec, s.y_tmp);
      float omega_denom = s.y_tmp->copy_to_host()[0];
      s.omega_val = omega_num / omega_denom;

      // x = x + alpha*p + omega*s
      s.d_ptr->saxpy(s.stream_id, s.n, s.alpha_val, s.p, s.x);
      s.d_ptr->saxpy(s.stream_id, s.n, s.omega_val, s.s_vec, s.x);

      // r = s - omega*t
      s.d_ptr->scopy(s.stream_id, s.n, s.s_vec, s.r);
      s.d_ptr->saxpy(s.stream_id, s.n, -s.omega_val, s.t_vec, s.r);

      // check convergence: norm_sq = <r, r>
      s.d_ptr->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
      norm_sq = s.y_tmp->copy_to_host()[0];
    }

    finish_solve();
  }

  void finish_solve() {
    auto& s = state();
    // Copy result back to host and notify supervisor
    std::vector<float> solution = s.x->copy_to_host();
    if (s.supervisor)
      this->mail(std::move(solution)).send(s.supervisor);
    this->quit();
  }

  void execute_spmv(mem_ptr<float> input_v, mem_ptr<float> output_v) {
    auto& s = state();
    switch (s.format) {
      case matrix_format::csr: s.d_ptr->spmv_csr(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v, s.spmv_workspace); break;
      case matrix_format::csc: s.d_ptr->spmv_csc(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v, s.spmv_workspace); break;
      case matrix_format::coo: s.d_ptr->spmv_coo(s.stream_id, s.n, s.n, s.nnz, 1.0f, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, 0.0f, output_v, s.spmv_workspace); break;
      default: break;
    }
  }
};

} // namespace caf::cuda