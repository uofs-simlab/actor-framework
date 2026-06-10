#pragma once

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <string>
#include <vector>
#include <memory>
#include "atoms.hpp"
#include "sparse_utils.hpp"
#include "caf/actorSOLVE/actorSOLVE.hpp"

using namespace caf;
using namespace caf::cuda;

// ---------------------------- FAULT TOLERANT SOLVER ----------------------------

template <class T = float>
struct ft_cg_state {
  // Host Data
  std::string path;
  in<int> h_row_ptr, h_col_ind;
  in<T> h_values, h_b;
  in_out<T> h_x;

  // GPU Buffers
  mem_ptr<int> A_rp, A_ci, d_err;
  mem_ptr<T> A_val, b, x, r, p, w, y_tmp;
  mem_ptr<char> spmv_ws;

  // Config
  int n, nnz, max_iter;
  int iterations = 0;
  int strikes = 0;
  T tol;
  int device_id, stream_id;

  // Supervision & Monitoring
  caf::actor supervisor;
  device_ptr d_ptr;
  program_ptr stab_prog;

  T initial_rho = 0;
  T current_rho = 0;
  T old_rho = 0;
  bool initialized = false;
  std::shared_ptr<MatrixData> pinned_data;
};

template <class T = float>
behavior fault_tolerant_cg_actor(stateful_actor<ft_cg_state<T>>* self,
                                 std::string path,
                                 std::shared_ptr<MatrixData> data,
                                 in<int> rp, in<int> ci, in<T> val, in<T> b_in, in_out<T> x_in,
                                 int n, int nnz, T tol, int max_iter,
                                 int dev_num, int stream, caf::actor supervisor) {
  auto& s = self->state();
  s.pinned_data = std::move(data);
  s.path = std::move(path);
  s.h_row_ptr = std::move(rp); s.h_col_ind = std::move(ci);
  s.h_values = std::move(val); s.h_b = std::move(b_in); s.h_x = std::move(x_in);
  s.n = n; s.nnz = nnz; s.tol = tol; s.max_iter = max_iter;
  s.device_id = dev_num; s.stream_id = stream; s.supervisor = supervisor;

  return {
    [=](start_atom) {
      auto& st = self->state();
      if (st.initialized) return;

      command_runner<> runner;
      st.A_rp = runner.transfer_memory(st.device_id, st.stream_id, st.h_row_ptr);
      st.A_ci = runner.transfer_memory(st.device_id, st.stream_id, st.h_col_ind);
      st.A_val = runner.transfer_memory(st.device_id, st.stream_id, st.h_values);
      st.b = runner.transfer_memory(st.device_id, st.stream_id, st.h_b);
      st.x = runner.transfer_memory(st.device_id, st.stream_id, st.h_x);

      st.d_ptr = platform::create()->schedule(st.stream_id, st.device_id);
      st.d_ptr->enable_cublas(); st.d_ptr->enable_cusparse();

      auto& mgr = manager::get();
      // Load stability kernel from file as requested
      st.stab_prog = mgr.create_program_from_cubin("stability_kernels.cubin", "check_stability", st.d_ptr);

      st.r = runner.transfer_memory(st.device_id, st.stream_id, out<T>(st.n));
      st.p = runner.transfer_memory(st.device_id, st.stream_id, out<T>(st.n));
      st.w = runner.transfer_memory(st.device_id, st.stream_id, out<T>(st.n));
      st.y_tmp = runner.transfer_memory(st.device_id, st.stream_id, out<T>(1));
      st.d_err = runner.transfer_memory(st.device_id, st.stream_id, out<int>(1));
      
      size_t ws_sz = st.d_ptr->spmv_csr_buffer_size(st.stream_id, st.n, st.n, st.nnz, st.A_rp, st.A_ci, st.A_val, st.x, st.w);
      if (ws_sz > 0) st.spmv_ws = command_runner<out<char>>{}.transfer_memory(st.device_id, st.stream_id, out<char>{static_cast<int>(ws_sz)});

      st.d_ptr->spmv_csr(st.stream_id, st.n, st.n, st.nnz, T{1}, st.A_rp, st.A_ci, st.A_val, st.x, T{0}, st.w, st.spmv_ws);
      if constexpr (std::is_same_v<T, double>) st.d_ptr->dcopy(st.stream_id, st.n, st.b, st.r);
      else st.d_ptr->scopy(st.stream_id, st.n, st.b, st.r);
      if constexpr (std::is_same_v<T, double>) st.d_ptr->daxpy(st.stream_id, st.n, -1.0, st.w, st.r);
      else st.d_ptr->saxpy(st.stream_id, st.n, -1.0f, st.w, st.r);
      if constexpr (std::is_same_v<T, double>) st.d_ptr->ddot(st.stream_id, st.n, st.r, st.r, st.y_tmp);
      else st.d_ptr->sdot(st.stream_id, st.n, st.r, st.r, st.y_tmp);

      st.initial_rho = runner.copy_to_host(st.y_tmp)[0];
      st.current_rho = st.initial_rho;
      st.initialized = true;

      // Notify supervisor that setup is complete and report initial status
      solver_result_meta meta(st.device_id, st.stream_id, 0, false, CG_SUCCESS);
      self->mail(gpu_done_atom_v, st.path, actor_cast<actor>(self), std::vector<T>{}, meta).send(st.supervisor);
    },

    [=](cg_next_step_atom, int num_iters) {
      auto& st = self->state();
      command_runner<T> runner;
      T threshold = st.tol * st.tol;
      int code = CG_SUCCESS;
      int step_count = 0;

      // Execute exactly the number of iterations requested by the supervisor
      while (step_count < num_iters && st.iterations < st.max_iter && st.current_rho > threshold) {
        // std::cout << "iterations = " << st.iterations << ", current_rho = " << st.current_rho << std::endl;
        st.iterations++;
        step_count++;

        if (st.iterations > 1) {
          T beta = st.current_rho / st.old_rho;
          if constexpr (std::is_same_v<T, double>) {
            st.d_ptr->dcopy(st.stream_id, st.n, st.r, st.w);
            st.d_ptr->daxpy(st.stream_id, st.n, beta, st.p, st.w);
            st.d_ptr->dcopy(st.stream_id, st.n, st.w, st.p);
          } else {
            st.d_ptr->scopy(st.stream_id, st.n, st.r, st.w);
            st.d_ptr->saxpy(st.stream_id, st.n, static_cast<float>(beta), st.p, st.w);
            st.d_ptr->scopy(st.stream_id, st.n, st.w, st.p);
          }
        } else {
          if constexpr (std::is_same_v<T, double>) st.d_ptr->dcopy(st.stream_id, st.n, st.r, st.p);
          else st.d_ptr->scopy(st.stream_id, st.n, st.r, st.p);
        }

        st.d_ptr->spmv_csr(st.stream_id, st.n, st.n, st.nnz, T{1}, st.A_rp, st.A_ci, st.A_val, st.p, T{0}, st.w, st.spmv_ws);
        if constexpr (std::is_same_v<T, double>) st.d_ptr->ddot(st.stream_id, st.n, st.p, st.w, st.y_tmp);
        else st.d_ptr->sdot(st.stream_id, st.n, st.p, st.w, st.y_tmp);
        
        T dot_pw = runner.copy_to_host(st.y_tmp)[0];
        if (std::abs(dot_pw) < 1e-25) { 
          code = CG_BREAKDOWN; 
          break; 
        }

        T alpha = st.current_rho / dot_pw;
        if constexpr (std::is_same_v<T, double>) {
          st.d_ptr->daxpy(st.stream_id, st.n, alpha, st.p, st.x);
          st.d_ptr->daxpy(st.stream_id, st.n, -alpha, st.w, st.r);
        } else {
          st.d_ptr->saxpy(st.stream_id, st.n, static_cast<float>(alpha), st.p, st.x);
          st.d_ptr->saxpy(st.stream_id, st.n, static_cast<float>(-alpha), st.w, st.r);
        }

        st.old_rho = st.current_rho;
        if constexpr (std::is_same_v<T, double>) st.d_ptr->ddot(st.stream_id, st.n, st.r, st.r, st.y_tmp);
        else st.d_ptr->sdot(st.stream_id, st.n, st.r, st.r, st.y_tmp);
        st.current_rho = runner.copy_to_host(st.y_tmp)[0];

      }

      // Run error checks and prepare progress report
      bool converged = (st.current_rho <= threshold);

      // Check for non-fatal errors that can be retried (Stagnation, Max Iter, Residual Factor)
      if (code == CG_SUCCESS && !converged && std::abs(st.old_rho - st.current_rho) < 1e-12)
        code = CG_STAGNATION;

      if (code == CG_SUCCESS) {
        if (!converged && st.iterations >= st.max_iter) code = CG_MAX_ITER;
        // Residual decrease check: treat as non-fatal strike if residual didn't decrease significantly
        if (st.initial_rho > 0 && (st.current_rho / st.initial_rho) > 0.999) code = CG_RESIDUAL_FACTOR_FAIL;
        
        // If we reached here with CG_SUCCESS, reset strikes as this was a productive batch
        if (code == CG_SUCCESS) st.strikes = 0;
      }

      // Reset and launch NaN/Inf stability kernel from file
      nd_range range(static_cast<int>((st.n + 255) / 256), 1, 1, 256, 1, 1);
      CHECK_CUDA(cuMemsetD32Async(st.d_err->mem(), 0, 1, st.d_ptr->get_stream_for_actor(st.stream_id)));
      st.d_ptr->launch_kernel_mem_ref(st.stab_prog->get_kernel(st.d_ptr->getId()), range,
                                      std::make_tuple(in<int>(st.n), st.x, st.r, st.d_err), st.stream_id);

      int err_flag = runner.copy_to_host(st.d_err)[0];
      if (err_flag != 0 || std::isnan(st.current_rho) || std::isinf(st.current_rho)) code = CG_NAN_INF;

      // Three strikes policy for non-fatal errors (Stagnation, Max Iterations, Residual Factor Failure)
      bool is_fatal = (code == CG_NAN_INF || code == CG_BREAKDOWN);
      if (code != CG_SUCCESS && !is_fatal) {
        st.strikes++;
        if (st.strikes < 3) {
          code = CG_SUCCESS; // Reset code to SUCCESS to allow the supervisor to retry/suspend
        }
      }

      if (code != CG_SUCCESS) converged = false;

      solver_result_meta meta(st.device_id, st.stream_id, st.iterations, converged, code);
      
      // Report current solution and metadata to the supervisor
      runner.copy_to_host_async(st.x, [=, supervisor = st.supervisor, path = st.path, self_h = actor_cast<actor>(self)](std::vector<T> sol) {
        anon_mail(gpu_done_atom_v, path, self_h, std::move(sol), meta).send(supervisor);
        if (converged || code != CG_SUCCESS)
          self->quit();
      });
    },
    [=](update_stream_atom, int new_stream) {
      self->state().stream_id = new_stream;
    },
    [=](shutdown_atom) {
      self->quit();
    }
  };
}