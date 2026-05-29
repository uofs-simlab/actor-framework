#pragma once

#include <caf/all.hpp>
#include <vector>
#include <utility>
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
  T tol;
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
  sparse_bicgstab_actor(actor_config& cfg, in<int> rp, in<int> ci, in<T> val,
                        in<T> b, in_out<T> x, matrix_format fmt, int n,
                        int nnz, T tol, int max_iter, int device_num,
                        int stream_id,
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

  ~sparse_bicgstab_actor() override = default;

  behavior make_behavior() override {
    return {
      [this](start_atom) {
        auto& s = this->state();
        if (!s.supervisor)
          s.supervisor = actor_cast<caf::actor>(this->current_sender());
        start_solve();
      },
      [this](gpu_done_atom, std::vector<T>& solution, solver_result_meta meta) {
        auto& s = this->state();
        if (s.supervisor)
          this->mail(std::move(solution), meta).send(s.supervisor);
        this->quit();
      }
    };
  }
private:
  void start_solve() {
    auto& s = this->state();
    command_runner<T> runner;

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
    T norm_sq = runner.copy_to_host(s.y_tmp)[0];

    s.iterations = 0;
    s.rho_val = T{1};
    s.alpha_val = T{1};
    s.omega_val = T{1};

    // BiCGSTAB Loop
    while (norm_sq > (s.tol * s.tol) && s.iterations < s.max_iter) {
      s.iterations++;

      // rho_new = <r_hat, r>
      execute_dot(s.r_hat, s.r, s.y_tmp);
      T rho_new = runner.copy_to_host(s.y_tmp)[0];

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
      T alpha_denom = runner.copy_to_host(s.y_tmp)[0];
      s.alpha_val = s.rho_val / alpha_denom;

      // s = r - alpha * v
      execute_copy(s.r, s.s_vec);
      execute_axpy(-s.alpha_val, s.v, s.s_vec);

      // Convergence check: Exit if the solution is reached after the alpha update
      execute_dot(s.s_vec, s.s_vec, s.y_tmp);
      if (runner.copy_to_host(s.y_tmp)[0] < (s.tol * s.tol)) {
        execute_axpy(s.alpha_val, s.p, s.x);
        break;
      }

      // t = As
      execute_spmv(s.s_vec, s.t_vec);

      // omega = <t, s> / <t, t>
      execute_dot(s.t_vec, s.s_vec, s.y_tmp);
      T omega_num = runner.copy_to_host(s.y_tmp)[0];
      execute_dot(s.t_vec, s.t_vec, s.y_tmp);
      T omega_denom = runner.copy_to_host(s.y_tmp)[0];
      s.omega_val = omega_num / omega_denom;

      // x = x + alpha*p + omega*s
      execute_axpy(s.alpha_val, s.p, s.x);
      execute_axpy(s.omega_val, s.s_vec, s.x);

      // r = s - omega*t
      execute_copy(s.s_vec, s.r);
      execute_axpy(-s.omega_val, s.t_vec, s.r);

      // check convergence: norm_sq = <r, r>
      execute_dot(s.r, s.r, s.y_tmp);
      norm_sq = runner.copy_to_host(s.y_tmp)[0];
    }

    auto self = actor_cast<actor>(this);
    solver_result_meta meta(s.device_num, s.stream_id, s.iterations, norm_sq <= (s.tol * s.tol));
    runner.copy_to_host_async(s.x, [self, meta](std::vector<T> solution) {
      anon_mail(gpu_done_atom_v, std::move(solution), meta).send(self);
    });
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
  sparse_bicgstab_facade(actor_config& cfg, uint32_t response_id)
    : event_based_actor(cfg), reply_id_(response_id) {}

  behavior make_behavior() override {
    return {
      // Mode 1: Return mem_ptr handles (GPU memory)
      [this](return_mem_ptr_atom,
             in<int> rp, in<int> ci, in<T> val,
             in<T> b_in, in_out<T> x_in,
             matrix_format fmt, int n, int nnz,
             T tol, int max_iter,
             int device_num, int stream_id) {

        auto [x, meta] = solve_core(rp, ci, val, b_in, x_in,
                                    fmt, n, nnz, tol, max_iter,
                                    device_num, stream_id);

        if (auto sender = actor_cast<actor>(this->current_sender())) {
          caf::anon_mail(reply_id_, std::move(x), meta).send(sender);
        }
      },

      // Mode 2: Return host data via mappings
      [this](std::vector<output_mapping> mappings,
             in<int> rp, in<int> ci, in<T> val,
             in<T> b_in, in_out<T> x_in,
             matrix_format fmt, int n, int nnz,
             T tol, int max_iter,
             int device_num, int stream_id) {

        auto [x, meta] = solve_core(rp, ci, val, b_in, x_in,
                                    fmt, n, nnz, tol, max_iter,
                                    device_num, stream_id);

        dispatch_result(std::move(mappings), std::move(x), n, meta);
      },

      // Mode 3: Default (return vector to sender)
      [this](in<int> rp, in<int> ci, in<T> val, in<T> b_in, in_out<T> x_in,
             matrix_format fmt, int n, int nnz, T tol, int max_iter,
             int device_num, int stream_id) {

        auto [x, meta] = solve_core(rp, ci, val, b_in, x_in,
                                    fmt, n, nnz, tol, max_iter,
                                    device_num, stream_id);

        dispatch_result({}, std::move(x), n, meta);
      }
    };
  }

protected:
  virtual std::pair<mem_ptr<T>, solver_result_meta> solve_core(in<int> rp, in<int> ci, in<T> val, in<T> b_in,
                                                               in_out<T> x_in, matrix_format fmt, int n,
                                                               int nnz, T tol, int max_iter,
                                                               int device_num, int stream_id) {
    command_runner<T> runner;
    auto res = runner.transfer_memory(device_num, stream_id,
                                      rp, ci, val, b_in, x_in);

    auto A_row_ptr = std::get<0>(res);
    auto A_col_ind = std::get<1>(res);
    auto A_values  = std::get<2>(res);
    auto b         = std::get<3>(res);
    auto x         = std::get<4>(res);

    auto d_ptr = platform::create()->schedule(stream_id, device_num);

    command_runner<out<T>> work_runner;
    auto r     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto r_hat = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto p     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto v     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto s_vec = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto t_vec = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto y_tmp = work_runner.transfer_memory(device_num, stream_id, out<T>(1));

    mem_ptr<char> spmv_workspace;
    std::size_t ws_size = 0;
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

    execute_spmv(x, v); 
    execute_copy(b, r);
    execute_axpy(T{-1}, v, r);
    execute_copy(r, r_hat);
    execute_dot(r, r, y_tmp);
    T norm_sq = runner.copy_to_host(y_tmp)[0];

    int iterations = 0;
    T rho_val = T{1}, alpha_val = T{1}, omega_val = T{1}, beta_val = T{0};

    while (norm_sq > (tol * tol) && iterations < max_iter) {
      iterations++;
      execute_dot(r_hat, r, y_tmp);
      T rho_new = runner.copy_to_host(y_tmp)[0];

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
      T alpha_denom = runner.copy_to_host(y_tmp)[0];
      alpha_val = rho_val / alpha_denom;
      execute_copy(r, s_vec);
      execute_axpy(-alpha_val, v, s_vec);
      execute_spmv(s_vec, t_vec);
      execute_dot(t_vec, s_vec, y_tmp);
      T omega_num = runner.copy_to_host(y_tmp)[0];
      execute_dot(t_vec, t_vec, y_tmp);
      T omega_denom = runner.copy_to_host(y_tmp)[0];
      omega_val = omega_num / omega_denom;
      execute_axpy(alpha_val, p, x);
      execute_axpy(omega_val, s_vec, x);
      execute_copy(s_vec, r);
      execute_axpy(-omega_val, t_vec, r);
      execute_dot(r, r, y_tmp);
      norm_sq = runner.copy_to_host(y_tmp)[0];
    }
    return {x, solver_result_meta(device_num, stream_id, iterations, norm_sq <= (tol * tol))};
  }

  void dispatch_result(std::vector<output_mapping> mappings,
                       mem_ptr<T> x,
                       int n,
                       solver_result_meta meta) {

    auto sender = actor_cast<actor>(this->current_sender());
    if (!sender) return;

    void* custom_dst = nullptr;
    size_t custom_count = 0;

    for (const auto& m : mappings) {
      if (m.index == 4) {
        custom_dst = m.dst;
        custom_count = m.count;
        break;
      }
    }

    command_runner<T> runner;

    if (custom_dst) {
      runner.copy_to_host_async(
        x,
        static_cast<T*>(custom_dst),
        custom_count > 0 ? custom_count : (size_t)n,
        [sender, r_id = reply_id_, meta](T*, size_t) {
          caf::anon_mail(r_id, 4, meta).send(sender);
        });

    } else {
      runner.copy_to_host_async(
        x,
        [sender, r_id = reply_id_, meta](std::vector<T> data) {
          caf::anon_mail(r_id, 4, std::move(data), meta).send(sender);
        });
    }
  }

  uint32_t reply_id_;
};

/**
 * A variant of the BiCGSTAB solver facade that uses Jacobi preconditioning.
 */
template <class T = float>
class sparse_bicgstab_jacobi_facade : public sparse_bicgstab_facade<T> {
public:
  sparse_bicgstab_jacobi_facade(actor_config& cfg, uint32_t response_id)
    : sparse_bicgstab_facade<T>(cfg, response_id) {
    // Deduce path to cubin relative to this header file at runtime
    std::string current_file = __FILE__;
    auto pos = current_file.find_last_of('/');
    std::string dir = (pos == std::string::npos) ? "" : current_file.substr(0, pos + 1);
    auto& mgr = manager::get();
    diag_prog_ = mgr.create_program_from_cubin(dir + "jacobi_kernels.cubin", "extract_diag_inv");
  }

protected:
  std::pair<mem_ptr<T>, solver_result_meta> solve_core(in<int> rp, in<int> ci, in<T> val, in<T> b_in,
                                                       in_out<T> x_in,
                                                       matrix_format fmt, int n, int nnz,
                                                       T tol, int max_iter,
                                                       int device_num, int stream_id) override {

    command_runner<T> runner;
    auto res = runner.transfer_memory(device_num, stream_id,
                                      rp, ci, val, b_in, x_in);

    auto A_row_ptr = std::get<0>(res);
    auto A_col_ind = std::get<1>(res);
    auto A_values  = std::get<2>(res);
    auto b         = std::get<3>(res);
    auto x         = std::get<4>(res);

    auto d_ptr = platform::create()->schedule(stream_id, device_num);

    command_runner<out<T>> work_runner;
    auto r     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto r_hat = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto p     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto v     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto s_vec = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto t_vec = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto D_inv = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto p_hat = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto s_hat = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto y_tmp = work_runner.transfer_memory(device_num, stream_id, out<T>(1));

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

    // Preconditioning setup: Extract diagonal inverse
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    nd_range range(blocks, 1, 1, threads, 1, 1);
    d_ptr->launch_kernel_mem_ref(diag_prog_->get_kernel(d_ptr->getId()), range,
                                 std::make_tuple(in<int>(n), A_row_ptr, A_col_ind, A_values, D_inv),
                                 stream_id);

    // Initial Residual: r = b - Ax
    execute_spmv(x, v); 
    execute_copy(b, r);
    execute_axpy(T{-1}, v, r);
    execute_copy(r, r_hat);
    execute_dot(r, r, y_tmp);
    T norm_sq = runner.copy_to_host(y_tmp)[0];

    int iterations = 0;
    T rho_val = T{1}, alpha_val = T{1}, omega_val = T{1}, beta_val = T{0};

    while (norm_sq > (tol * tol) && iterations < max_iter) {
      iterations++;
      execute_dot(r_hat, r, y_tmp);
      T rho_new = runner.copy_to_host(y_tmp)[0];

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

      // Apply Preconditioner: p_hat = D_inv * p
      if constexpr (std::is_same_v<T, float>)
        d_ptr->s_elementwise_multiply(stream_id, n, D_inv, p, p_hat);
      else
        d_ptr->d_elementwise_multiply(stream_id, n, D_inv, p, p_hat);

      execute_spmv(p_hat, v);
      execute_dot(r_hat, v, y_tmp);
      T alpha_denom = runner.copy_to_host(y_tmp)[0];
      alpha_val = rho_val / alpha_denom;
      execute_copy(r, s_vec);
      execute_axpy(-alpha_val, v, s_vec);
      
      // Convergence check: Exit if the solution is reached after the alpha update
      execute_dot(s_vec, s_vec, y_tmp);
      T s_norm_sq = runner.copy_to_host(y_tmp)[0];
      if (s_norm_sq <= (tol * tol)) {
        execute_axpy(alpha_val, p_hat, x);
        return {x, solver_result_meta(device_num, stream_id, iterations, true)};
      }

      // Apply Preconditioner: s_hat = D_inv * s
      if constexpr (std::is_same_v<T, float>)
        d_ptr->s_elementwise_multiply(stream_id, n, D_inv, s_vec, s_hat);
      else
        d_ptr->d_elementwise_multiply(stream_id, n, D_inv, s_vec, s_hat);

      execute_spmv(s_hat, t_vec);
      execute_dot(t_vec, s_vec, y_tmp); // Corrected: Numerator uses unpreconditioned residual s
      T omega_num = runner.copy_to_host(y_tmp)[0];
      execute_dot(t_vec, t_vec, y_tmp);
      T omega_denom = runner.copy_to_host(y_tmp)[0];
      omega_val = omega_num / omega_denom;
      execute_axpy(alpha_val, p_hat, x);
      execute_axpy(omega_val, s_hat, x);
      execute_copy(s_vec, r);
      execute_axpy(-omega_val, t_vec, r);
      execute_dot(r, r, y_tmp);
      norm_sq = runner.copy_to_host(y_tmp)[0];
    }
    return {x, solver_result_meta(device_num, stream_id, iterations, norm_sq <= (tol * tol))};
  }
private:
  program_ptr diag_prog_;
};

} // namespace caf::cuda