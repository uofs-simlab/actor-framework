#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <memory>
#include "caf/actorSOLVE/actorSOLVE.hpp"
#include "sparse_utils.hpp"

using namespace caf;
using namespace caf::cuda;
namespace fs = std::filesystem;

constexpr uint32_t WORKLOAD_SEED = 42;

template <class Inspector>
bool inspect(Inspector& f, SolverType& x) {
    auto val = static_cast<int>(x);
    if (f.apply(val)) {
        if constexpr (Inspector::is_loading)
            x = static_cast<SolverType>(val);
        return true;
    }
    return false;
}

CAF_BEGIN_TYPE_ID_BLOCK(workload_test, caf::id_block::cuda::end)
    CAF_ADD_ATOM(workload_test, get_work_atom)
    CAF_ADD_ATOM(workload_test, release_memory_atom)
    CAF_ADD_ATOM(workload_test, request_work_atom)
    CAF_ADD_ATOM(workload_test, worker_done_atom)
    CAF_ADD_ATOM(workload_test, work_tick_atom)
    CAF_ADD_ATOM(workload_test, add_work_atom)
    CAF_ADD_ATOM(workload_test, steal_work_atom)
    CAF_ADD_ATOM(workload_test, shutdown_atom)
    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<caf::actor>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))
CAF_END_TYPE_ID_BLOCK(workload_test)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixData)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)

const char* check_stability_src = R"(
extern "C" __global__
void check_stability(int n, const float* x, const float* r, int* err) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (isnan(x[idx]) || isinf(x[idx]) || isnan(r[idx]) || isinf(r[idx]))
      *err = 1;
  }
}
)";

/**
 * Error codes for solver_result_meta.error_code
 */
enum cg_error_type : int {
  CG_SUCCESS = 0,
  CG_MAX_ITER = 1,
  CG_NAN_INF = 2,
  CG_STAGNATION = 3,
  CG_BREAKDOWN = 4
  CG_RESIDUAL_FACTOR_FAIL = 5
};

// ---------------------------- FAULT TOLERANT SOLVER ----------------------------

template <class T = float>
struct ft_cg_state {
  // Host Data
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
  T tol;
  int device_id, stream_id;

  // Supervision & Monitoring
  caf::actor supervisor;
  device_ptr d_ptr;
  program_ptr stab_prog;

  T initial_rho = 0;
  T current_rho = 0;
};

template <class T = float>
behavior fault_tolerant_cg_actor(stateful_actor<ft_cg_state<T>>* self,
                                 in<int> rp, in<int> ci, in<T> val, in<T> b_in, in_out<T> x_in,
                                 int n, int nnz, T tol, int max_iter,
                                 int dev_num, int stream, caf::actor supervisor) {
  auto& s = self->state();
  s.h_row_ptr = std::move(rp); s.h_col_ind = std::move(ci);
  s.h_values = std::move(val); s.h_b = std::move(b_in); s.h_x = std::move(x_in);
  s.n = n; s.nnz = nnz; s.tol = tol; s.max_iter = max_iter;
  s.device_id = dev_num; s.stream_id = stream; s.supervisor = supervisor;

  return {
    [self](start_atom) {
      auto& st = self->state();
      command_runner<T> runner;

      // Setup and transfer memory
      auto res = runner.transfer_memory(st.device_id, st.stream_id, st.h_row_ptr, st.h_col_ind, st.h_values, st.h_b, st.h_x);
      st.A_rp = std::get<0>(res); st.A_ci = std::get<1>(res); st.A_val = std::get<2>(res);
      st.b = std::get<3>(res); st.x = std::get<4>(res);

      st.d_ptr = platform::create()->schedule(st.stream_id, st.device_id);
      st.d_ptr->enable_cublas(); st.d_ptr->enable_cusparse();
      auto& mgr = manager::get();
      st.stab_prog = mgr.create_program(check_stability_src, "check_stability", st.d_ptr);

      command_runner<out<T>> work_runner;
      st.r = work_runner.transfer_memory(st.device_id, st.stream_id, out<T>(st.n));
      st.p = work_runner.transfer_memory(st.device_id, st.stream_id, out<T>(st.n));
      st.w = work_runner.transfer_memory(st.device_id, st.stream_id, out<T>(st.n));
      st.y_tmp = work_runner.transfer_memory(st.device_id, st.stream_id, out<T>(1));
      st.d_err = command_runner<in_out<int>>{}.transfer_memory(st.device_id, st.stream_id, in_out<int>(0));

      // SpMV workspace allocation
      size_t ws_sz = st.d_ptr->spmv_csr_buffer_size(st.stream_id, st.n, st.n, st.nnz, st.A_rp, st.A_ci, st.A_val, st.x, st.w);
      if (ws_sz > 0) st.spmv_ws = command_runner<out<char>>{}.transfer_memory(st.device_id, st.stream_id, out<char>{static_cast<int>(ws_sz)});

      // Initial r = b - Ax, initial rho = r*r
      st.d_ptr->spmv_csr(st.stream_id, st.n, st.n, st.nnz, T{1}, st.A_rp, st.A_ci, st.A_val, st.x, T{0}, st.w, st.spmv_ws);
      if constexpr (std::is_same_v<T, double>) st.d_ptr->dcopy(st.stream_id, st.n, st.b, st.r);
      else st.d_ptr->scopy(st.stream_id, st.n, st.b, st.r);
      if constexpr (std::is_same_v<T, double>) st.d_ptr->daxpy(st.stream_id, st.n, -1.0, st.w, st.r);
      else st.d_ptr->saxpy(st.stream_id, st.n, -1.0f, st.w, st.r);
      if constexpr (std::is_same_v<T, double>) st.d_ptr->ddot(st.stream_id, st.n, st.r, st.r, st.y_tmp);
      else st.d_ptr->sdot(st.stream_id, st.n, st.r, st.r, st.y_tmp);

      st.initial_rho = runner.copy_to_host(st.y_tmp)[0];
      st.current_rho = st.initial_rho;
      T threshold = st.tol * st.tol;
      T old_rho = 0;
      int code = CG_SUCCESS;

      // SIMPLE TIGHT LOOP (Synchronous execution within handler to avoid overhead)
      while (st.iterations < st.max_iter && st.current_rho > threshold) {
        st.iterations++;
        if (st.iterations > 1) {
          T beta = st.current_rho / old_rho;
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
        
        if (std::abs(dot_pw) < 1e-25) { code = CG_BREAKDOWN; break; }
        
        T alpha = st.current_rho / dot_pw;
        if constexpr (std::is_same_v<T, double>) {
          st.d_ptr->daxpy(st.stream_id, st.n, alpha, st.p, st.x);
          st.d_ptr->daxpy(st.stream_id, st.n, -alpha, st.w, st.r);
        } else {
          st.d_ptr->saxpy(st.stream_id, st.n, static_cast<float>(alpha), st.p, st.x);
          st.d_ptr->saxpy(st.stream_id, st.n, static_cast<float>(-alpha), st.w, st.r);
        }
        old_rho = st.current_rho;
        if constexpr (std::is_same_v<T, double>) st.d_ptr->ddot(st.stream_id, st.n, st.r, st.r, st.y_tmp);
        else st.d_ptr->sdot(st.stream_id, st.n, st.r, st.r, st.y_tmp);
        st.current_rho = runner.copy_to_host(st.y_tmp)[0];
        
        if (std::abs(old_rho - st.current_rho) < 1e-18) { code = CG_STAGNATION; break; }
      }

      // POST-LOOP FAULT CHECKS
      bool converged = (st.current_rho <= threshold);
      if (code == CG_SUCCESS) {
        if (!converged) code = CG_MAX_ITER;
        // Residual decrease check: fail if residual didn't decrease by factor of 10
        if (st.initial_rho > 0 && (st.current_rho / st.initial_rho) > 0.1) code = CG_RESIDUAL_FACTOR_FAIL;
      }

      // Launch NaN/Inf stability kernel
      nd_range range((st.n + 255) / 256, 1, 1, 256, 1, 1);
      st.d_ptr->launch_kernel_mem_ref(st.stab_prog->get_kernel(st.d_ptr->getId()), range,
                                      std::make_tuple(in<int>(st.n), st.x, st.r, st.d_err), st.stream_id);
      
      int err_flag = runner.copy_to_host(st.d_err)[0];
      if (err_flag != 0 || std::isnan(st.current_rho) || std::isinf(st.current_rho)) code = CG_NAN_INF;

      solver_result_meta meta(st.device_id, st.stream_id, st.iterations, converged, code);
      runner.copy_to_host_async(st.x, [self, meta, supervisor = st.supervisor](std::vector<T> sol) {
        anon_mail(std::move(sol), meta).send(supervisor);
        self->quit();
      });
    }
  };
}



void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
   
    manager::shutdown();
}
CAF_MAIN(id_block::cuda,id_block::workload_test)