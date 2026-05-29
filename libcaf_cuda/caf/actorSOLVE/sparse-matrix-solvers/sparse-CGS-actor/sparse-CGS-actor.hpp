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
 device_ptr d_ptr;
 caf::actor supervisor;

 // Workspace vectors
 mem_ptr<float> r, p, w, y_tmp;
 mem_ptr<char> spmv_workspace;
 
 // Scalars needed across asynchronous steps
 float rho_val = 0.0f;
 float old_rho_val = 0.0f;
 float alpha_val = 0.0f;
 float beta_val = 0.0f;
 float dot_pw_val = 0.0f;
 int iterations = 0;
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
 }

 ~sparse_cg_actor() override = default;

 behavior make_behavior() override {
   return {
     [this](start_atom) {
       auto& s = state();
       if (!s.supervisor)
         s.supervisor = actor_cast<caf::actor>(this->current_sender());
       start_solve();
     },
     [this](gpu_done_atom, std::vector<float>& solution) {
       if (state().supervisor)
         this->mail(std::move(solution)).send(state().supervisor);
       this->quit();
     }
   };
 }

private:
 void start_solve() {
   auto& s = state();
   command_runner<float> runner;

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
   command_runner<out<float>> work_runner;
   s.r = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.p = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.w = work_runner.transfer_memory(s.device_num, s.stream_id, out<float>(s.n));
   s.y_tmp = work_runner.transfer_memory(s.device_num, s.stream_id, create_out_arg_with_size<float>(1));
   
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

   // 1. Initial SpMV: w = A * x
   execute_spmv(s.x, s.w);

   // 2. Initial r = b - w
   s.d_ptr->scopy(s.stream_id, s.n, s.b, s.r);
   s.d_ptr->saxpy(s.stream_id, s.n, -1.0f, s.w, s.r);

   // 3. Initial rho = r * r
   s.d_ptr->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
   
   // Fetch initial rho synchronously to start the loop
   s.rho_val = runner.copy_to_host(s.y_tmp)[0];

   // The Real Performance Fix: The Tight CG Loop
   // By running the loop here, we eliminate 20,000+ scheduler context switches.
   while (s.rho_val > (s.tol * s.tol) && s.iterations < s.max_iter) {
     s.iterations++;

     if (s.iterations > 1) {
       s.beta_val = s.rho_val / s.old_rho_val;
       s.d_ptr->scopy(s.stream_id, s.n, s.r, s.w);
       s.d_ptr->saxpy(s.stream_id, s.n, s.beta_val, s.p, s.w);
       s.d_ptr->scopy(s.stream_id, s.n, s.w, s.p);
     } else {
       s.d_ptr->scopy(s.stream_id, s.n, s.r, s.p);
     }

     execute_spmv(s.p, s.w);

     s.d_ptr->sdot(s.stream_id, s.n, s.p, s.w, s.y_tmp);
     // This synchronous call blocks the CAF thread ONLY until this dot product is ready.
     // This is 100x faster than yielding to the scheduler.
     s.dot_pw_val = runner.copy_to_host(s.y_tmp)[0];

     s.alpha_val = s.rho_val / s.dot_pw_val;
     s.d_ptr->saxpy(s.stream_id, s.n, s.alpha_val, s.p, s.x);
     s.d_ptr->saxpy(s.stream_id, s.n, -s.alpha_val, s.w, s.r);

     s.old_rho_val = s.rho_val;
     s.d_ptr->sdot(s.stream_id, s.n, s.r, s.r, s.y_tmp);
     s.rho_val = runner.copy_to_host(s.y_tmp)[0];
   }

   // Exit the loop and return the result via the standard async path
   auto self = actor_cast<actor>(this);
   runner.copy_to_host_async(s.x, [self](std::vector<float> solution) {
     anon_mail(gpu_done_atom_v, std::move(solution)).send(self);
   });
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

/**
 * A stateless (facade) variant of the CG solver.
 * This actor can be reused for multiple solve requests. It receives all 
 * solve parameters as a message and returns the solution vector to the sender.
 */
class sparse_cg_facade : public event_based_actor {
public:
  sparse_cg_facade(actor_config& cfg, uint32_t response_id)
    : event_based_actor(cfg), reply_id_(response_id) {}

  behavior make_behavior() override {
    return {
      // Mode 1: Return mem_ptr handles (GPU memory)
      [this](return_mem_ptr_atom,
             in<int> rp, in<int> ci, in<float> val,
             in<float> b_in, in_out<float> x_in,
             matrix_format fmt, int n, int nnz,
             float tol, int max_iter,
             int device_num, int stream_id) {

        auto x = solve_core(rp, ci, val, b_in, x_in,
                            fmt, n, nnz, tol, max_iter,
                            device_num, stream_id);

        if (auto sender = actor_cast<actor>(this->current_sender())) {
          caf::anon_mail(reply_id_, std::move(x)).send(sender);
        }
      },

      // Mode 2: Return host data via mappings
      [this](std::vector<output_mapping> mappings,
             in<int> rp, in<int> ci, in<float> val,
             in<float> b_in, in_out<float> x_in,
             matrix_format fmt, int n, int nnz,
             float tol, int max_iter,
             int device_num, int stream_id) {

        auto x = solve_core(rp, ci, val, b_in, x_in,
                            fmt, n, nnz, tol, max_iter,
                            device_num, stream_id);

        dispatch_result(std::move(mappings), std::move(x), n);
      },

      // Mode 3: Default (return vector to sender)
      [this](in<int> rp, in<int> ci, in<float> val,
             in<float> b_in, in_out<float> x_in,
             matrix_format fmt, int n, int nnz,
             float tol, int max_iter,
             int device_num, int stream_id) {

        auto x = solve_core(rp, ci, val, b_in, x_in,
                            fmt, n, nnz, tol, max_iter,
                            device_num, stream_id);

        dispatch_result({}, std::move(x), n);
      }
    };
  }

protected:
  virtual mem_ptr<float> solve_core(in<int> rp, in<int> ci, in<float> val, in<float> b_in,
                                    in_out<float> x_in,
                                    matrix_format fmt, int n, int nnz,
                                    float tol, int max_iter,
                                    int device_num, int stream_id) {

    command_runner<float> runner;

    auto res = runner.transfer_memory(device_num, stream_id,
                                      rp, ci, val, b_in, x_in);

    auto A_row_ptr = std::get<0>(res);
    auto A_col_ind = std::get<1>(res);
    auto A_values  = std::get<2>(res);
    auto b         = std::get<3>(res);
    auto x         = std::get<4>(res);

    auto d_ptr = platform::create()->schedule(stream_id, device_num);

    command_runner<out<float>> work_runner;
    auto r     = work_runner.transfer_memory(device_num, stream_id, out<float>(n));
    auto p     = work_runner.transfer_memory(device_num, stream_id, out<float>(n));
    auto w     = work_runner.transfer_memory(device_num, stream_id, out<float>(n));
    auto y_tmp = work_runner.transfer_memory(device_num, stream_id, create_out_arg_with_size<float>(1));

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

    auto execute_spmv = [&](mem_ptr<float> input_v, mem_ptr<float> output_v) {
      switch (fmt) {
        case matrix_format::csr:
          d_ptr->spmv_csr(stream_id, n, n, nnz, 1.0f,
                          A_row_ptr, A_col_ind, A_values,
                          input_v, 0.0f, output_v, spmv_workspace);
          break;

        case matrix_format::csc:
          d_ptr->spmv_csc(stream_id, n, n, nnz, 1.0f,
                          A_row_ptr, A_col_ind, A_values,
                          input_v, 0.0f, output_v, spmv_workspace);
          break;

        case matrix_format::coo:
          d_ptr->spmv_coo(stream_id, n, n, nnz, 1.0f,
                          A_row_ptr, A_col_ind, A_values,
                          input_v, 0.0f, output_v, spmv_workspace);
          break;

        default:
          break;
      }
    };

    execute_spmv(x, w);
    d_ptr->scopy(stream_id, n, b, r);
    d_ptr->saxpy(stream_id, n, -1.0f, w, r);
    d_ptr->sdot(stream_id, n, r, r, y_tmp);

    float rho_val = runner.copy_to_host(y_tmp)[0];
    float old_rho_val = 0.0f;
    int iterations = 0;

    while (rho_val > (tol * tol) && iterations < max_iter) {
      iterations++;

      if (iterations > 1) {
        float beta_val = rho_val / old_rho_val;
        d_ptr->scopy(stream_id, n, r, w);
        d_ptr->saxpy(stream_id, n, beta_val, p, w);
        d_ptr->scopy(stream_id, n, w, p);
      } else {
        d_ptr->scopy(stream_id, n, r, p);
      }

      execute_spmv(p, w);
      d_ptr->sdot(stream_id, n, p, w, y_tmp);

      float alpha_val =
        rho_val / runner.copy_to_host(y_tmp)[0];

      d_ptr->saxpy(stream_id, n, alpha_val, p, x);
      d_ptr->saxpy(stream_id, n, -alpha_val, w, r);

      old_rho_val = rho_val;
      d_ptr->sdot(stream_id, n, r, r, y_tmp);
      rho_val = runner.copy_to_host(y_tmp)[0];
    }

    return x;
  }

  void dispatch_result(std::vector<output_mapping> mappings,
                       mem_ptr<float> x,
                       int n) {

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

    command_runner<float> runner;

    if (custom_dst) {
      runner.copy_to_host_async(
        x,
        static_cast<float*>(custom_dst),
        custom_count > 0 ? custom_count : (size_t)n,
        [sender, r_id = reply_id_](float*, size_t) {
          caf::anon_mail(r_id, 4).send(sender);
        });

    } else {
      runner.copy_to_host_async(
        x,
        [sender, r_id = reply_id_](std::vector<float> data) {
          caf::anon_mail(r_id, 4, std::move(data)).send(sender);
        });
    }
  }

private:
  uint32_t reply_id_;
};

/**
 * A variant of the CG solver facade that uses Jacobi preconditioning.
 */
class sparse_cg_jacobi_facade : public sparse_cg_facade {
public:
  sparse_cg_jacobi_facade(actor_config& cfg, uint32_t response_id)
    : sparse_cg_facade(cfg, response_id) {
    // Deduce path to cubin relative to this header file at runtime
    std::string current_file = __FILE__;
    auto pos = current_file.find_last_of('/');
    std::string dir = (pos == std::string::npos) ? "" : current_file.substr(0, pos + 1);
    auto& mgr = manager::get();
    diag_prog_ = mgr.create_program_from_cubin(dir + "jacobi_kernels.cubin", "extract_diag_inv");
  }

protected:
  mem_ptr<float> solve_core(in<int> rp, in<int> ci, in<float> val, in<float> b_in,
                            in_out<float> x_in,
                            matrix_format fmt, int n, int nnz,
                            float tol, int max_iter,
                            int device_num, int stream_id) override {

    command_runner<float> runner;
    auto res = runner.transfer_memory(device_num, stream_id,
                                      rp, ci, val, b_in, x_in);

    auto A_row_ptr = std::get<0>(res);
    auto A_col_ind = std::get<1>(res);
    auto A_values  = std::get<2>(res);
    auto b         = std::get<3>(res);
    auto x         = std::get<4>(res);

    auto d_ptr = platform::create()->schedule(stream_id, device_num);

    command_runner<out<float>> work_runner;
    auto r     = work_runner.transfer_memory(device_num, stream_id, out<float>(n));
    auto p     = work_runner.transfer_memory(device_num, stream_id, out<float>(n));
    auto w     = work_runner.transfer_memory(device_num, stream_id, out<float>(n));
    auto z     = work_runner.transfer_memory(device_num, stream_id, out<float>(n));
    auto D_inv = work_runner.transfer_memory(device_num, stream_id, out<float>(n));
    auto y_tmp = work_runner.transfer_memory(device_num, stream_id, create_out_arg_with_size<float>(1));

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

    auto execute_spmv = [&](mem_ptr<float> input_v, mem_ptr<float> output_v) {
      switch (fmt) {
        case matrix_format::csr:
          d_ptr->spmv_csr(stream_id, n, n, nnz, 1.0f,
                          A_row_ptr, A_col_ind, A_values,
                          input_v, 0.0f, output_v, spmv_workspace);
          break;
        case matrix_format::csc:
          d_ptr->spmv_csc(stream_id, n, n, nnz, 1.0f,
                          A_row_ptr, A_col_ind, A_values,
                          input_v, 0.0f, output_v, spmv_workspace);
          break;
        case matrix_format::coo:
          d_ptr->spmv_coo(stream_id, n, n, nnz, 1.0f,
                          A_row_ptr, A_col_ind, A_values,
                          input_v, 0.0f, output_v, spmv_workspace);
          break;
        default:
          break;
      }
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
    d_ptr->scopy(stream_id, n, b, r);
    d_ptr->saxpy(stream_id, n, -1.0f, w, r);
    
    // 2. Initial Preconditioned Residual: z = D_inv * r
    d_ptr->s_elementwise_multiply(stream_id, n, D_inv, r, z);
    
    // 3. Initial rho = r * z
    d_ptr->sdot(stream_id, n, r, z, y_tmp);

    float rho_val = runner.copy_to_host(y_tmp)[0];
    float old_rho_val = 0.0f;
    int iterations = 0;

    while (rho_val > (tol * tol) && iterations < max_iter) {
      iterations++;

      if (iterations > 1) {
        float beta_val = rho_val / old_rho_val;
        // p = z + beta * p
        d_ptr->scopy(stream_id, n, z, w);
        d_ptr->saxpy(stream_id, n, beta_val, p, w);
        d_ptr->scopy(stream_id, n, w, p);
      } else {
        // p = z
        d_ptr->scopy(stream_id, n, z, p);
      }

      execute_spmv(p, w);
      d_ptr->sdot(stream_id, n, p, w, y_tmp);
      float alpha_val = rho_val / runner.copy_to_host(y_tmp)[0];

      d_ptr->saxpy(stream_id, n, alpha_val, p, x);
      d_ptr->saxpy(stream_id, n, -alpha_val, w, r);

      old_rho_val = rho_val;
      d_ptr->s_elementwise_multiply(stream_id, n, D_inv, r, z); // z_new = D_inv * r
      d_ptr->sdot(stream_id, n, r, z, y_tmp); // rho_new = r * z_new
      rho_val = runner.copy_to_host(y_tmp)[0];
    }

    return x;
  }

private:
  program_ptr diag_prog_;
};

} // namespace caf::cuda