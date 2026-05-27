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

template <class T = float>
struct sparse_bicgstab_state {
  // Host Data
  in<int> h_row_ptr, h_col_ind;
  in<T> h_values, h_b;
  in_out<T> h_x;
 
  // Device Problem data
  mem_ptr<int> A_row_ptr, A_col_ind;
  mem_ptr<T> A_values, b, x;
  matrix_format format;
  int n, nnz;
  float tol;
  int max_iter;
  int device_num;
  int stream_id;
  device_ptr d_ptr;
  caf::actor supervisor;

  // Workspace vectors
  mem_ptr<T> r, r_hat, p, v, s_vec, t_vec, y_tmp;
  mem_ptr<char> spmv_workspace;
  
  // Scalars needed across asynchronous steps
  T rho_val = T{1}; // Renamed from rho to avoid conflict with step-specific rho_new
  T alpha_val = T{1};
  T omega_val = T{1};
  T beta_val = T{0};
  int iterations = 0;
};

template <class T = float>
class sparse_bicgstab_actor : public stateful_actor<sparse_bicgstab_state<T>> {
public:
  sparse_bicgstab_actor(actor_config& cfg, in<int> rp, in<int> ci,
                        in<T> val, in<T> b, in_out<T> x,
                        matrix_format fmt, int n, int nnz, float tol, int max_iter, int device_num, int stream_id,
                        caf::actor supervisor = nullptr)
    : stateful_actor<sparse_bicgstab_state<T>>(cfg) {
    this->state().h_row_ptr = std::move(rp);
    this->state().h_col_ind = std::move(ci);
    this->state().h_values = std::move(val);
    this->state().h_b = std::move(b);
    this->state().h_x = std::move(x);
    this->state().format = fmt;
    this->state().n = n; this->state().nnz = nnz;
    this->state().tol = tol; this->state().max_iter = max_iter;
    this->state().device_num = device_num; this->state().stream_id = stream_id;
    this->state().supervisor = supervisor;
  }

  behavior make_behavior() override {
    return {
      [this](start_atom) {
        auto& s = this->state();
        if (!s.supervisor)
          s.supervisor = actor_cast<caf::actor>(this->current_sender());
        start_solve();
      },
    };
  }
private:
  void start_solve() {
    auto& s = this->state();
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
    command_runner<out<T>> work_runner;
    s.r     = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
    s.r_hat = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
    s.p     = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
    s.v     = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
    s.s_vec = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
    s.t_vec = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
    // Bottleneck Fix: Allocate only 1 scalar for results
    s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(1));

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
    execute_copy(s.b, s.r);
    execute_axpy(T{-1}, s.v, s.r); // r = b - Ax
    
    // 2. Choose r_hat = r
    execute_copy(s.r, s.r_hat);

    // 3. Initial norm calculation
    execute_dot(s.r, s.r, s.y_tmp);
    T norm_sq = s.y_tmp->copy_to_host()[0];

    s.iterations = 0;
    s.rho_val = T{1};
    s.alpha_val = T{1};
    s.omega_val = T{1};

    // BiCGSTAB Loop
    while (norm_sq > (s.tol * s.tol) && s.iterations < s.max_iter) {
      s.iterations++;

      // rho_new = <r_hat, r>
      execute_dot(s.r_hat, s.r, s.y_tmp);
      T rho_new = s.y_tmp->copy_to_host()[0];

      if (s.iterations == 1) {
        execute_copy(s.r, s.p);
      } else {
        s.beta_val = (rho_new / s.rho_val) * (s.alpha_val / s.omega_val);
        // p = r + beta * (p - omega * v)
        execute_axpy(-s.omega_val, s.v, s.p);
        execute_copy(s.r, s.s_vec);
        execute_axpy(s.beta_val, s.p, s.s_vec);
        execute_copy(s.s_vec, s.p);
      }
      s.rho_val = rho_new;

      // v = Ap
      execute_spmv(s.p, s.v);

      // alpha = rho / <r_hat, v>
      execute_dot(s.r_hat, s.v, s.y_tmp);
      T alpha_denom = s.y_tmp->copy_to_host()[0];
      s.alpha_val = s.rho_val / alpha_denom;

      // s = r - alpha * v
      execute_copy(s.r, s.s_vec);
      execute_axpy(-s.alpha_val, s.v, s.s_vec);

      // t = As
      execute_spmv(s.s_vec, s.t_vec);

      // omega = <t, s> / <t, t>
      execute_dot(s.t_vec, s.s_vec, s.y_tmp);
      T omega_num = s.y_tmp->copy_to_host()[0];
      execute_dot(s.t_vec, s.t_vec, s.y_tmp);
      T omega_denom = s.y_tmp->copy_to_host()[0];
      s.omega_val = omega_num / omega_denom;

      // x = x + alpha*p + omega*s
      execute_axpy(s.alpha_val, s.p, s.x);
      execute_axpy(s.omega_val, s.s_vec, s.x);

      // r = s - omega*t
      execute_copy(s.s_vec, s.r);
      execute_axpy(-s.omega_val, s.t_vec, s.r);

      // check convergence: norm_sq = <r, r>
      execute_dot(s.r, s.r, s.y_tmp);
      norm_sq = s.y_tmp->copy_to_host()[0];
    }

    finish_solve();
  }

  void finish_solve() {
    auto& s = this->state();
    // Copy result back to host and notify supervisor
    std::vector<T> solution = s.x->copy_to_host();
    if (s.supervisor)
      this->mail(std::move(solution)).send(s.supervisor);
    this->quit();
  }

  void execute_spmv(mem_ptr<T> input_v, mem_ptr<T> output_v) {
    auto& s = this->state();
    switch (s.format) {
      case matrix_format::csr: s.d_ptr->spmv_csr(s.stream_id, s.n, s.n, s.nnz, T{1}, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, T{0}, output_v, s.spmv_workspace); break;
      case matrix_format::csc: s.d_ptr->spmv_csc(s.stream_id, s.n, s.n, s.nnz, T{1}, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, T{0}, output_v, s.spmv_workspace); break;
      case matrix_format::coo: s.d_ptr->spmv_coo(s.stream_id, s.n, s.n, s.nnz, T{1}, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, T{0}, output_v, s.spmv_workspace); break;
      default: break;
    }
  }

  // Precision-aware dispatch helpers
  void execute_copy(mem_ptr<T> src, mem_ptr<T> dst) {
    auto& s = this->state();
    if constexpr (std::is_same_v<T, double>) {
      s.d_ptr->dcopy(s.stream_id, s.n, src, dst);
    } else {
      s.d_ptr->scopy(s.stream_id, s.n, src, dst);
    }
  }

  void execute_axpy(T alpha, mem_ptr<T> x, mem_ptr<T> y) {
    auto& s = this->state();
    if constexpr (std::is_same_v<T, double>) {
      s.d_ptr->daxpy(s.stream_id, s.n, alpha, x, y);
    } else {
      s.d_ptr->saxpy(s.stream_id, s.n, static_cast<float>(alpha), x, y);
    }
  }

  void execute_dot(mem_ptr<T> x, mem_ptr<T> y, mem_ptr<T> res) {
    auto& s = this->state();
    if constexpr (std::is_same_v<T, double>) {
      s.d_ptr->ddot(s.stream_id, s.n, x, y, res);
    } else {
      s.d_ptr->sdot(s.stream_id, s.n, x, y, res);
    }
  }

  void execute_nrm2(mem_ptr<T> x, mem_ptr<T> res) {
    auto& s = this->state();
    if constexpr (std::is_same_v<T, double>) {
      s.d_ptr->dnrm2(s.stream_id, s.n, x, res);
    } else {
      s.d_ptr->snrm2(s.stream_id, s.n, x, res);
    }
  }
};

/**
 * A stateless (facade) variant of the BiCGSTAB solver.
 * This actor can be reused for multiple solve requests. It receives all 
 * solve parameters as a message and returns the solution vector to the sender.
 */
template <class T = float>
class sparse_bicgstab_facade : public event_based_actor {
public:
  sparse_bicgstab_facade(actor_config& cfg) : event_based_actor(cfg) {}

  behavior make_behavior() override {
    return {
      [this](in<int> rp, in<int> ci, in<T> val, in<T> b_in, in_out<T> x_in,
             matrix_format fmt, int n, int nnz, float tol, int max_iter,
             int device_num, int stream_id) -> std::vector<T> {
        command_runner<> runner;

        // 1. Transfer problem data to device
        auto res = runner.transfer_memory(device_num, stream_id,
                                          rp, ci, val, b_in, x_in);

        auto A_row_ptr = std::get<0>(res);
        auto A_col_ind = std::get<1>(res);
        auto A_values  = std::get<2>(res);
        auto b         = std::get<3>(res);
        auto x         = std::get<4>(res);

        auto d_ptr = platform::create()->schedule(stream_id, device_num);

        // 2. Allocate workspace
        command_runner<out<T>> work_runner;
        auto r     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
        auto r_hat = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
        auto p     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
        auto v     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
        auto s_vec = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
        auto t_vec = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
        auto y_tmp = work_runner.transfer_memory(device_num, stream_id, out<T>(1));

        // 3. Setup SPMV workspace
        mem_ptr<char> spmv_workspace;
        size_t ws_size = 0;
        if (fmt == matrix_format::csr)
          ws_size = d_ptr->spmv_csr_buffer_size(stream_id, n, n, nnz, A_row_ptr, A_col_ind, A_values, x, v);
        else if (fmt == matrix_format::csc)
          ws_size = d_ptr->spmv_csc_buffer_size(stream_id, n, n, nnz, A_row_ptr, A_col_ind, A_values, x, v);
        else if (fmt == matrix_format::coo)
          ws_size = d_ptr->spmv_coo_buffer_size(stream_id, n, n, nnz, A_row_ptr, A_col_ind, A_values, x, v);

        if (ws_size > 0) {
          command_runner<out<char>> ws_runner;
          spmv_workspace = ws_runner.transfer_memory(device_num, stream_id, out<char>(static_cast<int>(ws_size)));
        }

        // Helper lambdas for GPU operations
        auto execute_spmv = [&](mem_ptr<T> input_v, mem_ptr<T> output_v) {
          switch (fmt) {
            case matrix_format::csr: d_ptr->spmv_csr(stream_id, n, n, nnz, T{1}, A_row_ptr, A_col_ind, A_values, input_v, T{0}, output_v, spmv_workspace); break;
            case matrix_format::csc: d_ptr->spmv_csc(stream_id, n, n, nnz, T{1}, A_row_ptr, A_col_ind, A_values, input_v, T{0}, output_v, spmv_workspace); break;
            case matrix_format::coo: d_ptr->spmv_coo(stream_id, n, n, nnz, T{1}, A_row_ptr, A_col_ind, A_values, input_v, T{0}, output_v, spmv_workspace); break;
            default: break;
          }
        };
        auto execute_copy = [&](mem_ptr<T> src, mem_ptr<T> dst) {
          if constexpr (std::is_same_v<T, double>) d_ptr->dcopy(stream_id, n, src, dst);
          else d_ptr->scopy(stream_id, n, src, dst);
        };
        auto execute_axpy = [&](T alpha, mem_ptr<T> ax, mem_ptr<T> ay) {
          if constexpr (std::is_same_v<T, double>) d_ptr->daxpy(stream_id, n, alpha, ax, ay);
          else d_ptr->saxpy(stream_id, n, static_cast<float>(alpha), ax, ay);
        };
        auto execute_dot = [&](mem_ptr<T> dx, mem_ptr<T> dy, mem_ptr<T> dres) {
          if constexpr (std::is_same_v<T, double>) d_ptr->ddot(stream_id, n, dx, dy, dres);
          else d_ptr->sdot(stream_id, n, dx, dy, dres);
        };

        // 4. Initial Residual: r = b - Ax
        execute_spmv(x, v); 
        execute_copy(b, r);
        execute_axpy(T{-1}, v, r);
        execute_copy(r, r_hat);
        execute_dot(r, r, y_tmp);
        T norm_sq = y_tmp->copy_to_host()[0];

        int iterations = 0;
        T rho_val = T{1}, alpha_val = T{1}, omega_val = T{1}, beta_val = T{0};

        // BiCGSTAB Loop
        while (norm_sq > (tol * tol) && iterations < max_iter) {
          iterations++;
          execute_dot(r_hat, r, y_tmp);
          T rho_new = y_tmp->copy_to_host()[0];

          if (iterations == 1) {
            execute_copy(r, p);
          } else {
            beta_val = (rho_new / rho_val) * (alpha_val / omega_val);
            execute_axpy(-omega_val, v, p);
            execute_copy(r, s_vec);
            execute_axpy(beta_val, p, s_vec);
            execute_copy(s_vec, p);
          }
          rho_val = rho_new;
          execute_spmv(p, v);
          execute_dot(r_hat, v, y_tmp);
          T alpha_denom = y_tmp->copy_to_host()[0];
          alpha_val = rho_val / alpha_denom;
          execute_copy(r, s_vec);
          execute_axpy(-alpha_val, v, s_vec);
          execute_spmv(s_vec, t_vec);
          execute_dot(t_vec, s_vec, y_tmp);
          T omega_num = y_tmp->copy_to_host()[0];
          execute_dot(t_vec, t_vec, y_tmp);
          T omega_denom = y_tmp->copy_to_host()[0];
          omega_val = omega_num / omega_denom;
          execute_axpy(alpha_val, p, x);
          execute_axpy(omega_val, s_vec, x);
          execute_copy(s_vec, r);
          execute_axpy(-omega_val, t_vec, r);
          execute_dot(r, r, y_tmp);
          norm_sq = y_tmp->copy_to_host()[0];
        }
        return x->copy_to_host();
      }
    };
  }
};

} // namespace caf::cuda