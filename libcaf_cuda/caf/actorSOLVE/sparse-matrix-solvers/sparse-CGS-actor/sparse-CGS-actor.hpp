#pragma once

#include <caf/all.hpp>
#include <deque>
#include <vector>
#include "caf/cuda/all.hpp"
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"
#include "caf/actorSPARSE/spmv-actor/spmv-actor.hpp"
#include "caf/actorBLAS/axpy-actor/axpy-actor.hpp"
#include "caf/actorBLAS/copy-actor/copy-actor.hpp"
#include "caf/cuda/platform.hpp"

namespace caf::cuda {


constexpr int id_dot = 100;
constexpr int id_spmv = 200;
constexpr int id_axpy = 300;
constexpr int id_copy = 400;

template <class T = float>
struct sparse_cg_state {
 // Host Data (stored until start)
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
 mem_ptr<T> r, p, w, y_tmp;
 mem_ptr<char> spmv_workspace;
 
 // Scalars needed across asynchronous steps
 T rho_val = T{0};
 T old_rho_val = T{0};
 T alpha_val = T{0};
 T beta_val = T{0};
 T dot_pw_val = T{0};
 int iterations = 0;
};

template <class T = float>
class sparse_cg_actor : public stateful_actor<sparse_cg_state<T>> {
public:
 sparse_cg_actor(actor_config& cfg, in<int> rp, in<int> ci,
                 in<T> val, in<T> b, in_out<T> x,
                 matrix_format fmt, int n, int nnz, T tol, int max_iter, int device_num, int stream_id,
                 caf::actor supervisor = nullptr)
   : stateful_actor<sparse_cg_state<T>>(cfg) {
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

 ~sparse_cg_actor() override = default;

 behavior make_behavior() override {
   return {
     [this](start_atom) {
       auto& s = this->state();
       if (!s.supervisor)
         s.supervisor = actor_cast<caf::actor>(this->current_sender());
       start_solve();
     },
     [this](gpu_done_atom, std::vector<T>& solution, solver_result_meta meta) {
       if (this->state().supervisor)
         this->mail(std::move(solution), meta).send(this->state().supervisor);
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

   s.d_ptr = platform::create()->schedule(s.stream_id, s.device_num);

   // Allocate workspace
   command_runner<out<T>> work_runner;
   s.r = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
   s.p = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
   s.w = work_runner.transfer_memory(s.device_num, s.stream_id, out<T>(s.n));
   s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, create_out_arg_with_size<T>(1));
   
   // Allocate SPMV workspace to avoid reallocations in the loop
   size_t ws_size = 0;
   if (s.format == matrix_format::csr)
     ws_size = s.d_ptr->spmv_csr_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.w);
   else if (s.format == matrix_format::csc)
     ws_size = s.d_ptr->spmv_csc_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.w);
   else if (s.format == matrix_format::coo)
     ws_size = s.d_ptr->spmv_coo_buffer_size(s.stream_id, s.n, s.n, s.nnz, s.A_row_ptr, s.A_col_ind, s.A_values, s.x, s.w);

   if (ws_size > 0) {
     command_runner<out<char>> ws_runner;
     s.spmv_workspace = ws_runner.transfer_memory(s.device_num, s.stream_id, out<char>(static_cast<int>(ws_size)));
   }

   auto execute_copy = [&](mem_ptr<T> src, mem_ptr<T> dst) {
     if constexpr (std::is_same_v<T, double>) s.d_ptr->dcopy(s.stream_id, s.n, src, dst); else s.d_ptr->scopy(s.stream_id, s.n, src, dst);
   };
   auto execute_axpy = [&](T alpha, mem_ptr<T> xv, mem_ptr<T> yv) {
     if constexpr (std::is_same_v<T, double>) s.d_ptr->daxpy(s.stream_id, s.n, alpha, xv, yv); else s.d_ptr->saxpy(s.stream_id, s.n, static_cast<float>(alpha), xv, yv);
   };
   auto execute_dot = [&](mem_ptr<T> xv, mem_ptr<T> yv, mem_ptr<T> rv) {
     if constexpr (std::is_same_v<T, double>) s.d_ptr->ddot(s.stream_id, s.n, xv, yv, rv); else s.d_ptr->sdot(s.stream_id, s.n, xv, yv, rv);
   };

   // 1. Initial SpMV: w = A * x
   execute_spmv(s.x, s.w);

   // 2. Initial r = b - w
   execute_copy(s.b, s.r);
   execute_axpy(T{-1}, s.w, s.r);

   // 3. Initial rho = r * r
   execute_dot(s.r, s.r, s.y_tmp);
   
   // Fetch initial rho synchronously to start the loop
   s.rho_val = runner.copy_to_host(s.y_tmp)[0];
   T threshold = s.tol * s.tol;

   // The Real Performance Fix: The Tight CG Loop
   // By running the loop here, we eliminate 20,000+ scheduler context switches.
   while (s.rho_val > threshold && s.iterations < s.max_iter) {
     s.iterations++;

     if (s.iterations > 1) {
       s.beta_val = s.rho_val / s.old_rho_val;
       execute_copy(s.r, s.w);
       execute_axpy(s.beta_val, s.p, s.w);
       execute_copy(s.w, s.p);
     } else {
       execute_copy(s.r, s.p);
     }

     execute_spmv(s.p, s.w);

     execute_dot(s.p, s.w, s.y_tmp);
     // This synchronous call blocks the CAF thread ONLY until this dot product is ready.
     // This is 100x faster than yielding to the scheduler.
     s.dot_pw_val = runner.copy_to_host(s.y_tmp)[0];

     s.alpha_val = s.rho_val / s.dot_pw_val;
     execute_axpy(s.alpha_val, s.p, s.x);
     execute_axpy(-s.alpha_val, s.w, s.r);

     s.old_rho_val = s.rho_val;
     execute_dot(s.r, s.r, s.y_tmp);
     s.rho_val = runner.copy_to_host(s.y_tmp)[0];
   }

   // Exit the loop and return the result via the standard async path
   auto self = actor_cast<actor>(this);
   solver_result_meta meta(s.device_num, s.stream_id, s.iterations, s.rho_val <= threshold);
   runner.copy_to_host_async(s.x, [self, meta](std::vector<T> solution) {
     anon_mail(gpu_done_atom_v, std::move(solution), meta).send(self);
   });
 }

 void execute_spmv(mem_ptr<T> input_v, mem_ptr<T> output_v) {
   auto& s = this->state();
   switch (s.format) {
     case matrix_format::csr:
       s.d_ptr->spmv_csr(s.stream_id, s.n, s.n, s.nnz, T{1}, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, T{0}, output_v, s.spmv_workspace);
       break;
     case matrix_format::csc:
       s.d_ptr->spmv_csc(s.stream_id, s.n, s.n, s.nnz, T{1}, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, T{0}, output_v, s.spmv_workspace);
       break;
     case matrix_format::coo:
       s.d_ptr->spmv_coo(s.stream_id, s.n, s.n, s.nnz, T{1}, s.A_row_ptr, s.A_col_ind, s.A_values, input_v, T{0}, output_v, s.spmv_workspace);
       break;
     default: break;
   }
 }
};

/**
 * A stateless (facade) variant of the CG solver.
 * This actor can be reused for multiple solve requests. It receives all 
 * solve parameters as a message and returns the solution vector to the sender.
 */
template <class T = float>
class sparse_cg_facade : public event_based_actor {
public:
  sparse_cg_facade(actor_config& cfg, uint32_t response_id)
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

        dispatch_result(actor_cast<caf::actor>(this->current_sender()), std::move(mappings), std::move(x), n, meta);
      },

      // Mode 3: Default (return vector to sender)
      [this](in<int> rp, in<int> ci, in<T> val,
             in<T> b_in, in_out<T> x_in,
             matrix_format fmt, int n, int nnz,
             T tol, int max_iter,
             int device_num, int stream_id) {

        auto [x, meta] = solve_core(rp, ci, val, b_in, x_in,
                                    fmt, n, nnz, tol, max_iter,
                                    device_num, stream_id);

        dispatch_result(actor_cast<caf::actor>(this->current_sender()), {}, std::move(x), n, meta);
      },

      // Mode 4: mem_ptr variant (assumes data is already on the GPU)
      [this](mem_ptr<int> rp, mem_ptr<int> ci, mem_ptr<T> val,
             mem_ptr<T> b_in, mem_ptr<T> x_in,
             matrix_format fmt, int n, int nnz,
             T tol, int max_iter,
             int device_num, int stream_id) {
        auto [x, meta] = solve_core_mem_ptr(rp, ci, val, b_in, x_in,
                                            fmt, n, nnz, tol, max_iter,
                                            device_num, stream_id);
        dispatch_result(actor_cast<caf::actor>(this->current_sender()), {}, std::move(x), n, meta);
      }
    };
  }

protected:
  virtual std::pair<mem_ptr<T>, solver_result_meta> solve_core(in<int> rp, in<int> ci, in<T> val, in<T> b_in,
                                                                   in_out<T> x_in,
                                                                   matrix_format fmt, int n, int nnz,
                                                                   T tol, int max_iter,
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
    auto p     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto w     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto y_tmp = work_runner.transfer_memory(device_num, stream_id, create_out_arg_with_size<T>(1));

    mem_ptr<char> spmv_workspace;
    size_t ws_size = 0;

    if (fmt == matrix_format::csr)
      ws_size = d_ptr->spmv_csr_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);
    else if (fmt == matrix_format::csc)
      ws_size = d_ptr->spmv_csc_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);
    else if (fmt == matrix_format::coo)
      ws_size = d_ptr->spmv_coo_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);

    if (ws_size > 0) {
      command_runner<out<char>> ws_runner;
      spmv_workspace =
        ws_runner.transfer_memory(device_num, stream_id,
                                  out<char>(static_cast<int>(ws_size)));
    }

    auto execute_spmv = [&](mem_ptr<T> input_v, mem_ptr<T> output_v) {
      switch (fmt) {
        case matrix_format::csr:
          d_ptr->spmv_csr(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;

        case matrix_format::csc:
          d_ptr->spmv_csc(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;

        case matrix_format::coo:
          d_ptr->spmv_coo(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;

        default:
          break;
      }
    };
    auto execute_copy = [&](mem_ptr<T> src, mem_ptr<T> dst) {
      if constexpr (std::is_same_v<T, double>) d_ptr->dcopy(stream_id, n, src, dst); else d_ptr->scopy(stream_id, n, src, dst);
    };
    auto execute_axpy = [&](T alpha, mem_ptr<T> xv, mem_ptr<T> yv) {
      if constexpr (std::is_same_v<T, double>) d_ptr->daxpy(stream_id, n, alpha, xv, yv); else d_ptr->saxpy(stream_id, n, static_cast<float>(alpha), xv, yv);
    };
    auto execute_dot = [&](mem_ptr<T> xv, mem_ptr<T> yv, mem_ptr<T> rv) {
      if constexpr (std::is_same_v<T, double>) d_ptr->ddot(stream_id, n, xv, yv, rv); else d_ptr->sdot(stream_id, n, xv, yv, rv);
    };

    execute_spmv(x, w);
    execute_copy(b, r);
    execute_axpy(T{-1}, w, r);
    execute_dot(r, r, y_tmp);

    T rho_val = runner.copy_to_host(y_tmp)[0];
    T old_rho_val = T{0};
    int iterations = 0;
    T threshold = tol * tol;

    while (rho_val > threshold && iterations < max_iter) {
      iterations++;

      if (iterations > 1) {
        T beta_val = rho_val / old_rho_val;
        execute_copy(r, w);
        execute_axpy(beta_val, p, w);
        execute_copy(w, p);
      } else {
       execute_copy(r, p);
      }

      execute_spmv(p, w);
      execute_dot(p, w, y_tmp);

      T alpha_val =
        rho_val / runner.copy_to_host(y_tmp)[0];

      execute_axpy(alpha_val, p, x);
      execute_axpy(-alpha_val, w, r);

      old_rho_val = rho_val;
      execute_dot(r, r, y_tmp);
      rho_val = runner.copy_to_host(y_tmp)[0];
    }

    return {x, solver_result_meta(device_num, stream_id, iterations, rho_val <= threshold)};
  }

  // New solve_core overload that accepts mem_ptr directly.
  // This bypasses the initial transfer_memory call as the data is assumed
  // to already be on the device and managed by the provided mem_ptrs.
  virtual std::pair<mem_ptr<T>, solver_result_meta> solve_core_mem_ptr(mem_ptr<int> A_row_ptr, mem_ptr<int> A_col_ind, mem_ptr<T> A_values, mem_ptr<T> b,
                                                                   mem_ptr<T> x,
                                                                   matrix_format fmt, int n, int nnz,
                                                                   T tol, int max_iter,
                                                                   int device_num, int stream_id) {

    command_runner<T> runner;

    auto d_ptr = platform::create()->schedule(stream_id, device_num);

    command_runner<out<T>> work_runner;
    auto r     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto p     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto w     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto y_tmp = work_runner.transfer_memory(device_num, stream_id, create_out_arg_with_size<T>(1));

    mem_ptr<char> spmv_workspace;
    size_t ws_size = 0;

    if (fmt == matrix_format::csr)
      ws_size = d_ptr->spmv_csr_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);
    else if (fmt == matrix_format::csc)
      ws_size = d_ptr->spmv_csc_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);
    else if (fmt == matrix_format::coo)
      ws_size = d_ptr->spmv_coo_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);

    if (ws_size > 0) {
      command_runner<out<char>> ws_runner;
      spmv_workspace =
        ws_runner.transfer_memory(device_num, stream_id,
                                  out<char>(static_cast<int>(ws_size)));
    }

    auto execute_spmv = [&](mem_ptr<T> input_v, mem_ptr<T> output_v) {
      switch (fmt) {
        case matrix_format::csr:
          d_ptr->spmv_csr(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;

        case matrix_format::csc:
          d_ptr->spmv_csc(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;

        case matrix_format::coo:
          d_ptr->spmv_coo(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;

        default:
          break;
      }
    };
    auto execute_copy = [&](mem_ptr<T> src, mem_ptr<T> dst) {
      if constexpr (std::is_same_v<T, double>) d_ptr->dcopy(stream_id, n, src, dst); else d_ptr->scopy(stream_id, n, src, dst);
    };
    auto execute_axpy = [&](T alpha, mem_ptr<T> xv, mem_ptr<T> yv) {
      if constexpr (std::is_same_v<T, double>) d_ptr->daxpy(stream_id, n, alpha, xv, yv); else d_ptr->saxpy(stream_id, n, static_cast<float>(alpha), xv, yv);
    };
    auto execute_dot = [&](mem_ptr<T> xv, mem_ptr<T> yv, mem_ptr<T> rv) {
      if constexpr (std::is_same_v<T, double>) d_ptr->ddot(stream_id, n, xv, yv, rv); else d_ptr->sdot(stream_id, n, xv, yv, rv);
    };

    execute_spmv(x, w);
    execute_copy(b, r);
    execute_axpy(T{-1}, w, r);
    execute_dot(r, r, y_tmp);

    T rho_val = runner.copy_to_host(y_tmp)[0];
    T old_rho_val = T{0};
    int iterations = 0;
    T threshold = tol * tol;

    while (rho_val > threshold && iterations < max_iter) {
      iterations++;

      if (iterations > 1) {
        T beta_val = rho_val / old_rho_val;
        execute_copy(r, w);
        execute_axpy(beta_val, p, w);
        execute_copy(w, p);
      } else {
       execute_copy(r, p);
      }

      execute_spmv(p, w);
      execute_dot(p, w, y_tmp);

      T alpha_val =
        rho_val / runner.copy_to_host(y_tmp)[0];

      execute_axpy(alpha_val, p, x);
      execute_axpy(-alpha_val, w, r);

      old_rho_val = rho_val;
      execute_dot(r, r, y_tmp);
      rho_val = runner.copy_to_host(y_tmp)[0];
    }

    return {x, solver_result_meta(device_num, stream_id, iterations, rho_val <= threshold)};
  }

  void dispatch_result(caf::actor target,
                       std::vector<output_mapping> mappings,
                       mem_ptr<T> x,
                       int n,
                       solver_result_meta meta) {
    if (!target) return;

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
        [target, r_id = reply_id_, meta](T*, size_t) {
          caf::anon_mail(r_id, 4, meta).send(target);
        });

    } else {
      runner.copy_to_host_async(
        x,
        [target, r_id = reply_id_, meta](std::vector<T> data) {
          caf::anon_mail(r_id, 4, std::move(data), meta).send(target);
        });
    }
  }

protected:
  uint32_t reply_id_;
};

/**
 * A variant of the CG solver facade that uses Jacobi preconditioning.
 */
template <class T = float>
class sparse_cg_jacobi_facade : public sparse_cg_facade<T> {
public:
  sparse_cg_jacobi_facade(actor_config& cfg, uint32_t response_id)
    : sparse_cg_facade<T>(cfg, response_id) {
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
    auto p     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto w     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto z     = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto D_inv = work_runner.transfer_memory(device_num, stream_id, out<T>(n));
    auto y_tmp = work_runner.transfer_memory(device_num, stream_id, create_out_arg_with_size<T>(1));

    mem_ptr<char> spmv_workspace;
    size_t ws_size = 0;

    if (fmt == matrix_format::csr)
      ws_size = d_ptr->spmv_csr_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);
    else if (fmt == matrix_format::csc)
      ws_size = d_ptr->spmv_csc_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);
    else if (fmt == matrix_format::coo)
      ws_size = d_ptr->spmv_coo_buffer_size(stream_id, n, n, nnz,
                                            A_row_ptr, A_col_ind, A_values, x, w);

    if (ws_size > 0) {
      command_runner<out<char>> ws_runner;
      spmv_workspace =
        ws_runner.transfer_memory(device_num, stream_id,
                                  out<char>(static_cast<int>(ws_size)));
    }

    auto execute_spmv = [&](mem_ptr<T> input_v, mem_ptr<T> output_v) {
      switch (fmt) {
        case matrix_format::csr:
          d_ptr->spmv_csr(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;
        case matrix_format::csc:
          d_ptr->spmv_csc(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;
        case matrix_format::coo:
          d_ptr->spmv_coo(stream_id, n, n, nnz, T{1},
                          A_row_ptr, A_col_ind, A_values,
                          input_v, T{0}, output_v, spmv_workspace);
          break;
        default:
          break;
      }
    };
    auto execute_copy = [&](mem_ptr<T> src, mem_ptr<T> dst) {
      if constexpr (std::is_same_v<T, double>) d_ptr->dcopy(stream_id, n, src, dst); else d_ptr->scopy(stream_id, n, src, dst);
    };
    auto execute_axpy = [&](T alpha, mem_ptr<T> xv, mem_ptr<T> yv) {
      if constexpr (std::is_same_v<T, double>) d_ptr->daxpy(stream_id, n, alpha, xv, yv); else d_ptr->saxpy(stream_id, n, static_cast<float>(alpha), xv, yv);
    };
    auto execute_dot = [&](mem_ptr<T> xv, mem_ptr<T> yv, mem_ptr<T> rv) {
      if constexpr (std::is_same_v<T, double>) d_ptr->ddot(stream_id, n, xv, yv, rv); else d_ptr->sdot(stream_id, n, xv, yv, rv);
    };

    // 0. Preconditioning setup: Extract diagonal inverse using custom kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    nd_range range(blocks, 1, 1, threads, 1, 1);

    d_ptr->launch_kernel_mem_ref(diag_prog_->get_kernel(d_ptr->getId()), range,
                                 std::make_tuple(in<int>(n), A_row_ptr, A_col_ind, A_values, D_inv),
                                 stream_id);

    // 1. Initial Residual: r = b - Ax
    execute_spmv(x, w);
    execute_copy(b, r);
    execute_axpy(T{-1}, w, r);
    
    // 2. Initial Preconditioned Residual: z = D_inv * r
    if constexpr (std::is_same_v<T, double>) d_ptr->d_elementwise_multiply(stream_id, n, D_inv, r, z); else d_ptr->s_elementwise_multiply(stream_id, n, D_inv, r, z);
    
    // 3. Initial rho = r * z
    execute_dot(r, z, y_tmp);

    T rho_val = runner.copy_to_host(y_tmp)[0];
    T old_rho_val = T{0};
    int iterations = 0;
    T threshold = tol * tol;

    while (rho_val > threshold && iterations < max_iter) {
      iterations++;

      if (iterations > 1) {
        T beta_val = rho_val / old_rho_val;
        // p = z + beta * p
        execute_copy(z, w);
        execute_axpy(beta_val, p, w);
        execute_copy(w, p);
      } else {
        // p = z
        execute_copy(z, p);
      }

      execute_spmv(p, w);
      execute_dot(p, w, y_tmp);
      T alpha_val = rho_val / runner.copy_to_host(y_tmp)[0];

      execute_axpy(alpha_val, p, x);
      execute_axpy(-alpha_val, w, r);

      old_rho_val = rho_val;
      if constexpr (std::is_same_v<T, double>) d_ptr->d_elementwise_multiply(stream_id, n, D_inv, r, z); else d_ptr->s_elementwise_multiply(stream_id, n, D_inv, r, z);
      execute_dot(r, z, y_tmp); // rho_new = r * z_new
      rho_val = runner.copy_to_host(y_tmp)[0];
    }

    return {x, solver_result_meta(device_num, stream_id, iterations, rho_val <= threshold)};
  }

private:
  program_ptr diag_prog_;
};

/**
 * Optimized CG Facade.
 * - No host recurrence: alpha/beta updates happen on GPU.
 * - Message-based iteration: Solves every 10 iterations via self-callbacks.
 * - Fully asynchronous: Does not block CAF worker threads.
 */
template <class T = float>
class sparse_cg_facade_optimized : public sparse_cg_facade<T> {
public:
  using solve_context = sparse_cg_solve_context<T>;

  sparse_cg_facade_optimized(actor_config& cfg, uint32_t response_id)
    : sparse_cg_facade<T>(cfg, response_id) {
    auto& mgr = manager::get();
    std::string current_file = __FILE__;
    auto pos = current_file.find_last_of('/');
    std::string dir = (pos == std::string::npos) ? "" : current_file.substr(0, pos + 1);
    
    std::string p_name = std::is_same_v<T, double> ? "update_p_double" : "update_p_float";
    std::string xr_name = std::is_same_v<T, double> ? "update_x_r_double" : "update_x_r_float";

    update_p_prog_ = mgr.create_program_from_cubin(dir + "cg_solver_kernels.cubin", p_name.c_str());
    update_xr_prog_ = mgr.create_program_from_cubin(dir + "cg_solver_kernels.cubin", xr_name.c_str());
  }

  behavior make_behavior() override {
    return {
      [this](return_mem_ptr_atom, in<int> rp, in<int> ci, in<T> val, in<T> b, in_out<T> x,
             matrix_format fmt, int n, int nnz, T tol, int max_iter, int dev, int stream) {
        auto ctx = setup_solve(rp, ci, val, b, x, fmt, n, nnz, tol, max_iter, dev, stream);
        ctx->return_mem_ptr = true;
        launch_iterations(ctx);
      },

      [this](std::vector<output_mapping> mappings, in<int> rp, in<int> ci, in<T> val, in<T> b, in_out<T> x,
             matrix_format fmt, int n, int nnz, T tol, int max_iter, int dev, int stream) {
        auto ctx = setup_solve(rp, ci, val, b, x, fmt, n, nnz, tol, max_iter, dev, stream);
        ctx->mappings = std::move(mappings);
        launch_iterations(ctx);
      },
      
      // Mode 3: Default (return vector to sender) - now handled by optimized facade
      [this](in<int> rp, in<int> ci, in<T> val,
             in<T> b_in, in_out<T> x_in,
             matrix_format fmt, int n, int nnz,
             T tol, int max_iter,
             int device_num, int stream_id) {
        auto ctx = setup_solve(rp, ci, val, b_in, x_in, fmt, n, nnz, tol, max_iter, device_num, stream_id);
        launch_iterations(ctx);
      },

      [this](next_batch_atom, std::shared_ptr<solve_context> ctx, T current_rho) {
        if (current_rho <= ctx->threshold || ctx->iterations >= ctx->max_iter) {
          solver_result_meta meta(ctx->device_num, ctx->stream_id, ctx->iterations, current_rho <= ctx->threshold);
          if (ctx->return_mem_ptr) {
            caf::anon_mail(this->reply_id_, std::move(ctx->x), meta).send(ctx->requester);
          } else {
            this->dispatch_result(ctx->requester, std::move(ctx->mappings), ctx->x, ctx->n, meta);
          }
        } else {
          launch_iterations(ctx);
        }
      }
    };
  }

protected:
  std::shared_ptr<solve_context> setup_solve(in<int> rp, in<int> ci, in<T> val, in<T> b_in, in_out<T> x_in,
                                            matrix_format fmt, int n, int nnz, T tol, int max_iter,
                                            int dev, int stream) {
    auto ctx = std::make_shared<solve_context>();
    ctx->requester = actor_cast<actor>(this->current_sender());
    ctx->n = n; ctx->nnz = nnz; ctx->max_iter = max_iter; ctx->threshold = tol * tol;
    ctx->device_num = dev; ctx->stream_id = stream; ctx->format = fmt;

    command_runner<T> runner;
    auto res = runner.transfer_memory(dev, stream, rp, ci, val, b_in, x_in);
    ctx->A_rp = std::get<0>(res); ctx->A_ci = std::get<1>(res); ctx->A_val = std::get<2>(res);
    ctx->b = std::get<3>(res); ctx->x = std::get<4>(res);

    command_runner<out<T>> work_runner;
    ctx->r = work_runner.transfer_memory(dev, stream, out<T>(n));
    ctx->p = work_runner.transfer_memory(dev, stream, out<T>(n));
    ctx->w = work_runner.transfer_memory(dev, stream, out<T>(n));
    ctx->rho = work_runner.transfer_memory(dev, stream, create_out_arg_with_size<T>(1));
    ctx->old_rho = work_runner.transfer_memory(dev, stream, create_out_arg_with_size<T>(1));
    ctx->dot_pw = work_runner.transfer_memory(dev, stream, create_out_arg_with_size<T>(1));

    auto d_ptr = platform::create()->schedule(stream, dev);

    size_t ws_size = 0;
    if (fmt == matrix_format::csr)
      ws_size = d_ptr->spmv_csr_buffer_size(stream, n, n, nnz, ctx->A_rp, ctx->A_ci, ctx->A_val, ctx->x, ctx->w);
    else if (fmt == matrix_format::csc)
      ws_size = d_ptr->spmv_csc_buffer_size(stream, n, n, nnz, ctx->A_rp, ctx->A_ci, ctx->A_val, ctx->x, ctx->w);
    else if (fmt == matrix_format::coo)
      ws_size = d_ptr->spmv_coo_buffer_size(stream, n, n, nnz, ctx->A_rp, ctx->A_ci, ctx->A_val, ctx->x, ctx->w);

    if (ws_size > 0) {
      command_runner<out<char>> ws_runner;
      ctx->spmv_ws = ws_runner.transfer_memory(dev, stream, out<char>(static_cast<int>(ws_size)));
    }

    // Initial r = b - Ax
    execute_spmv(ctx, ctx->x, ctx->w);
    if constexpr (std::is_same_v<T, double>) d_ptr->dcopy(stream, n, ctx->b, ctx->r); else d_ptr->scopy(stream, n, ctx->b, ctx->r);
    if constexpr (std::is_same_v<T, double>) d_ptr->daxpy(stream, n, -1.0, ctx->w, ctx->r); else d_ptr->saxpy(stream, n, -1.0f, ctx->w, ctx->r);
    execute_dot(ctx, ctx->r, ctx->r, ctx->rho);

    return ctx;
  }

  void launch_iterations(std::shared_ptr<solve_context> ctx) {
    auto d_ptr = platform::create()->schedule(ctx->stream_id, ctx->device_num);
    nd_range range((ctx->n + 255) / 256, 1, 1, 256, 1, 1);
    
    auto p_kernel = update_p_prog_->get_kernel(d_ptr->getId());
    auto xr_kernel = update_xr_prog_->get_kernel(d_ptr->getId());

    int batch_size = std::min(10, ctx->max_iter - ctx->iterations);
    for (int k = 0; k < batch_size; ++k) {
      // p = r + beta * p
      d_ptr->launch_kernel_mem_ref(p_kernel, range, 
                                   std::make_tuple(in<int>(ctx->n), ctx->r, ctx->p, ctx->rho, ctx->old_rho, in<int>(ctx->iterations), in<T>(ctx->threshold)), 
                                   ctx->stream_id);
      // w = Ap
      execute_spmv(ctx, ctx->p, ctx->w);
      // dot_pw = p * w
      execute_dot(ctx, ctx->p, ctx->w, ctx->dot_pw);
      // x += alpha * p, r -= alpha * w
      d_ptr->launch_kernel_mem_ref(xr_kernel, range,
                                   std::make_tuple(in<int>(ctx->n), ctx->x, ctx->r, ctx->p, ctx->w, ctx->rho, ctx->dot_pw, in<T>(ctx->threshold)),
                                   ctx->stream_id);
      // old_rho = rho, rho = r * r
      if constexpr (std::is_same_v<T, double>) d_ptr->dcopy(ctx->stream_id, 1, ctx->rho, ctx->old_rho); 
      else d_ptr->scopy(ctx->stream_id, 1, ctx->rho, ctx->old_rho);
      
      execute_dot(ctx, ctx->r, ctx->r, ctx->rho);
      ctx->iterations++;
    }

    // Asynchronous notification instead of blocking copy_to_host
    auto self = actor_cast<actor>(this);
    command_runner<T> runner;
    runner.copy_to_host_async(ctx->rho, [self, ctx](std::vector<T> res) {
      anon_mail(next_batch_atom_v, ctx, res[0]).send(self);
    });
  }

  void execute_spmv(std::shared_ptr<solve_context> ctx, mem_ptr<T> in_v, mem_ptr<T> out_v) {
    auto d_ptr = platform::create()->schedule(ctx->stream_id, ctx->device_num);
    switch (ctx->format) {
      case matrix_format::csr:
        d_ptr->spmv_csr(ctx->stream_id, ctx->n, ctx->n, ctx->nnz, T{1}, 
                        ctx->A_rp, ctx->A_ci, ctx->A_val, in_v, T{0}, out_v, ctx->spmv_ws);
        break;

      case matrix_format::csc:
        d_ptr->spmv_csc(ctx->stream_id, ctx->n, ctx->n, ctx->nnz, T{1}, 
                        ctx->A_rp, ctx->A_ci, ctx->A_val, in_v, T{0}, out_v, ctx->spmv_ws);
        break;

      case matrix_format::coo:
        d_ptr->spmv_coo(ctx->stream_id, ctx->n, ctx->n, ctx->nnz, T{1}, 
                        ctx->A_rp, ctx->A_ci, ctx->A_val, in_v, T{0}, out_v, ctx->spmv_ws);
        break;

      default:
        break;
    }
  }

  void execute_dot(std::shared_ptr<solve_context> ctx, mem_ptr<T> xv, mem_ptr<T> yv, mem_ptr<T> rv) {
    auto d_ptr = platform::create()->schedule(ctx->stream_id, ctx->device_num);
    if constexpr (std::is_same_v<T, double>) d_ptr->ddot(ctx->stream_id, ctx->n, xv, yv, rv); 
    else d_ptr->sdot(ctx->stream_id, ctx->n, xv, yv, rv);
  }

private:
  program_ptr update_p_prog_;
  program_ptr update_xr_prog_;
};

} // namespace caf::cuda