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
    CAF_ADD_TYPE_ID(workload_test, (MatrixData))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))
CAF_END_TYPE_ID_BLOCK(workload_test)



CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixData)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(SolverType)


/**
 * Error codes for solver_result_meta.error_code
 */
enum cg_error_type : int {
  CG_SUCCESS = 0,
  CG_MAX_ITER = 1,
  CG_NAN_INF = 2,
  CG_STAGNATION = 3,
  CG_BREAKDOWN = 4,
  CG_RESIDUAL_FACTOR_FAIL = 5
};

std::string to_string(cg_error_type err) {
  switch (err) {
    case CG_SUCCESS: return "Success";
    case CG_MAX_ITER: return "Maximum Iterations Reached";
    case CG_NAN_INF: return "Stability Check Failed (NaN/Inf Detected)";
    case CG_STAGNATION: return "Stagnation Detected (Residual stopped changing)";
    case CG_BREAKDOWN: return "Solver Breakdown (Division by zero/near-zero)";
    case CG_RESIDUAL_FACTOR_FAIL: return "Residual Factor Check Failed";
    default: return "Unknown Error (" + std::to_string(static_cast<int>(err)) + ")";
  }
}

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
  T old_rho = 0;
  bool initialized = false;
  std::shared_ptr<MatrixData> pinned_data;
};

template <class T = float>
behavior fault_tolerant_cg_actor(stateful_actor<ft_cg_state<T>>* self,
                                 std::shared_ptr<MatrixData> data,
                                 in<int> rp, in<int> ci, in<T> val, in<T> b_in, in_out<T> x_in,
                                 int n, int nnz, T tol, int max_iter,
                                 int dev_num, int stream, caf::actor supervisor) {
  auto& s = self->state();
  s.pinned_data = std::move(data);
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
      self->mail(gpu_done_atom_v, std::vector<T>{}, meta).send(st.supervisor);
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

        if (std::abs(st.old_rho - st.current_rho) < 1e-12) { 
          code = CG_STAGNATION; 
          break; 
        }
      }

      // Run error checks and prepare progress report
      bool converged = (st.current_rho <= threshold);
      if (code == CG_SUCCESS) {
        if (!converged && st.iterations >= st.max_iter) code = CG_MAX_ITER;
        // Residual decrease check: fail if residual didn't decrease significantly
        if (st.initial_rho > 0 && (st.current_rho / st.initial_rho) > 0.999) code = CG_RESIDUAL_FACTOR_FAIL;
      }

      // Reset and launch NaN/Inf stability kernel from file
      nd_range range(static_cast<int>((st.n + 255) / 256), 1, 1, 256, 1, 1);
      CHECK_CUDA(cuMemsetD32Async(st.d_err->mem(), 0, 1, st.d_ptr->get_stream_for_actor(st.stream_id)));
      st.d_ptr->launch_kernel_mem_ref(st.stab_prog->get_kernel(st.d_ptr->getId()), range,
                                      std::make_tuple(in<int>(st.n), st.x, st.r, st.d_err), st.stream_id);

      int err_flag = runner.copy_to_host(st.d_err)[0];
      if (err_flag != 0 || std::isnan(st.current_rho) || std::isinf(st.current_rho)) code = CG_NAN_INF;

      solver_result_meta meta(st.device_id, st.stream_id, st.iterations, converged, code);
      
      // Report current solution and metadata to the supervisor
      runner.copy_to_host_async(st.x, [=, supervisor = st.supervisor](std::vector<T> sol) {
        self->mail(gpu_done_atom_v, std::move(sol), meta).send(supervisor);
        if (converged || code != CG_SUCCESS) self->quit();
      });
    }
  };
}

// ---------------------------- SUPERVISOR ACTOR ----------------------------

struct supervisor_state {
    std::deque<MatrixTask> queue;
    std::vector<caf::actor> active_solvers;
    std::unordered_map<caf::actor_id, std::string> task_names;
    std::unordered_map<caf::actor_id, std::chrono::steady_clock::time_point> start_times;
    int max_active = 4; // Admission control limit to be mindful of GPU memory. If Illegal memory access that means two or more actors are using the same stream
    int num_iterations = 50;
    int stream = 0;
    int device = 0;
};

behavior supervisor_actor(stateful_actor<supervisor_state>* self, std::vector<MatrixTask> tasks) {
    auto& st = self->state();
    for (auto& t : tasks)
        st.queue.push_back(std::move(t));

    auto spawn_next = [self]() {
        auto& s = self->state();
        while (s.active_solvers.size() < static_cast<size_t>(s.max_active) && !s.queue.empty()) {
            auto task = std::move(s.queue.front());
            s.queue.pop_front();
            std::string path = task.path;
            self->println("[SUPE] Starting solver for: {}", path);
            auto solver = self->spawn(fault_tolerant_cg_actor<float>,
                                    task.data,
                                    create_in_arg(task.data->row_ptr),
                                    create_in_arg(task.data->col_indices),
                                    create_in_arg(task.data->values),
                                    create_in_arg(task.data->b),
                                    create_in_out_arg(task.data->x_guess),
                                    (int)task.data->row_ptr.size() - 1,
                                    (int)task.data->values.size(),
                                    1e-5f, 128000, s.device, (++s.stream)%32, actor_cast<actor>(self));
            s.task_names[solver->id()] = std::move(path);
            
            s.start_times[solver->id()] = std::chrono::steady_clock::now();
            s.active_solvers.push_back(solver);
            self->mail(start_atom_v).send(solver);
        }
    };

    spawn_next();
    
    return {
        [=](gpu_done_atom, std::vector<float>& solution, solver_result_meta meta) {
            auto& s = self->state();
            auto solver = actor_cast<caf::actor>(self->current_sender());
            
            if (meta.iterations == 0 && meta.error_code == CG_SUCCESS) {
                // Initialization report: trigger the first batch
                self->mail(cg_next_step_atom_v, s.num_iterations).send(solver);
                return;
            }

            std::string task_name = s.task_names.count(solver->id()) 
                                    ? s.task_names[solver->id()] 
                                    : "Unknown Task";

            if (meta.converged || meta.error_code != CG_SUCCESS) {
                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  end_time - s.start_times[solver->id()]).count();

                if (meta.converged) {
                    if (meta.iterations == 0)
                        self->println("[DONE] {}: Initial guess satisfied tolerance ({} ms).", task_name, duration);
                    else
                        self->println("[DONE] {}: Converged in {} iterations ({} ms).", task_name, meta.iterations, duration);
                } else {
                    self->println("[FAIL] {}: {} (after {} iterations, {} ms).", 
                                  task_name, 
                                  to_string(static_cast<cg_error_type>(meta.error_code)), 
                                  meta.iterations,
                                  duration);
                }

                auto it = std::find(s.active_solvers.begin(), s.active_solvers.end(), solver);
                if (it != s.active_solvers.end())
                    s.active_solvers.erase(it);
                s.task_names.erase(solver->id());
                s.start_times.erase(solver->id());

                spawn_next();

                if (s.active_solvers.empty() && s.queue.empty()) {
                    self->println("All tasks in the batch have been processed.");
                    self->quit();
                }
            } else {
                // Progress report: order the next iteration batch
                self->mail(cg_next_step_atom_v, s.num_iterations).send(solver);
            }
        }
    };
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
    std::cout << "loading\n";
    {
         //auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/spd", CGS_SOLVER);
        //auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/unsymmetric", CGS_SOLVER);
         //auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/unsymmetric", CGS_SOLVER);
         auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrices/mixed", CGS_SOLVER);

        std::cout << "loaded\n";
        if (tasks_vec.empty()) {
            std::cerr << "No matrices found. Running dummy test task." << std::endl;
            auto data = std::make_shared<MatrixData>();
            data->row_ptr = {0, 1, 2};
            data->col_indices = {0, 1};
            data->values = {10.0f, 10.0f};
            data->b = {100.0f, 100.0f};
            data->x_guess = {0.0f, 0.0f};
            tasks_vec.push_back({"dummy_task", CGS_SOLVER, data});
        }

        auto benchmark_start = std::chrono::steady_clock::now();

        sys.spawn(supervisor_actor, std::move(tasks_vec));
        sys.await_all_actors_done();

        auto benchmark_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> total_time = benchmark_end - benchmark_start;
        int num_gpus = manager::get().get_num_devices();

        std::cout << "\n";
        std::cout << "=====================================\n";
        std::cout << "IRREGULAR WORKLOAD BENCHMARK (CAF)\n";
        std::cout << "=====================================\n";
        std::cout << "Seed:               " << WORKLOAD_SEED << "\n";
        std::cout << "GPUs:               " << num_gpus << "\n";
        std::cout << "Admission Control:  " << 4 << "\n"; 
        std::cout << "Total Runtime:      " << total_time.count() << " s\n";
        std::cout << "=====================================\n";
    }
    manager::shutdown();
}
CAF_MAIN(id_block::cuda, id_block::workload_test)