#pragma once

#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <deque>
#include <unordered_map>
#include <vector>
#include <chrono>
#include "atoms.hpp"
#include "ft_cg_actor.hpp"
#include "sparse_utils.hpp"


using namespace caf;
using namespace caf::cuda;
// ---------------------------- SUPERVISOR ACTOR ----------------------------

struct suspended_task {
  caf::actor solver;
  std::string path;
  int last_batch_size;
  int device_id;
  int stream_id;
};

struct resource_slot {
  int device_id;
  int stream_id;
};

struct supervisor_state {
  std::deque<MatrixTask> queue;
  std::deque<suspended_task> suspended_queue;
  std::vector<caf::actor> active_solvers;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> start_times;
  std::unordered_map<std::string, resource_slot> task_resources;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> enqueue_times;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> pick_times;
  std::unordered_map<std::string, double> cumulative_active_ms;
  std::unordered_map<caf::actor, int> actor_batch_sizes;
  std::deque<resource_slot> available_slots;
  int max_active = 1; // Admission control limit to be mindful of GPU memory.
  int num_iterations = MAX_ITERATIONS;
  int num_gpus = 0;
  int tasks_succeeded = 0;
  int tasks_failed = 0;
  std::chrono::steady_clock::time_point benchmark_start;
};

behavior supervisor_actor(stateful_actor<supervisor_state>* self, std::vector<MatrixTask> tasks, 
  int initial_max_active, std::chrono::steady_clock::time_point start_time) {
    auto& st = self->state();
    st.queue.insert(st.queue.end(), std::make_move_iterator(tasks.begin()), std::make_move_iterator(tasks.end()));
    st.max_active = initial_max_active;
    st.benchmark_start = start_time;
    st.num_gpus = manager::get().get_num_devices();

    // Initialize the pool with streams interleaved across all available GPUs
    for (int s = 0; s < 4; ++s) {
      for (int g = 0; g < st.num_gpus; ++g) {
        st.available_slots.push_back({g, s});
      }
    }

    auto spawn_next = [self]() {
        auto& s = self->state();
        while (s.active_solvers.size() < static_cast<size_t>(s.max_active) && !s.available_slots.empty()) {
          if (!s.queue.empty()) {
            auto task = std::move(s.queue.front());
            s.queue.pop_front();
            std::string path = task.path;

            s.cumulative_active_ms[path] = 0;
            s.enqueue_times[path] = task.enqueue_time;
            s.pick_times[path] = std::chrono::steady_clock::now();

            resource_slot slot = s.available_slots.front();
            s.available_slots.pop_front();

            self->println("[INFO] Starting solver for: {} (Device: {}, Stream: {})", 
                          path, slot.device_id, slot.stream_id);
            auto solver = self->spawn(fault_tolerant_cg_actor<float>,
                                    path,
                                    task.data,
                                    create_in_arg(task.data->row_ptr),
                                    create_in_arg(task.data->col_indices),
                                    create_in_arg(task.data->values),
                                    create_in_arg(task.data->b),
                                    create_in_out_arg(task.data->x_guess),
                                    (int)task.data->row_ptr.size() - 1,
                                    (int)task.data->values.size(),
                                    1e-5f, MAX_ITERATIONS, slot.device_id, slot.stream_id, actor_cast<actor>(self));

            s.start_times[path] = std::chrono::steady_clock::now();
            s.active_solvers.push_back(solver);
            s.task_resources[path] = slot;
            s.actor_batch_sizes[solver] = s.num_iterations;
            self->mail(start_atom_v).send(solver);
          } else {
            bool resumed = false;
            for (auto it = s.suspended_queue.begin(); it != s.suspended_queue.end(); ++it) {
              auto slot_it = std::find_if(s.available_slots.begin(), s.available_slots.end(), [&](const resource_slot& slot) {
                return slot.device_id == it->device_id && slot.stream_id == it->stream_id;
              });
              if (slot_it != s.available_slots.end()) {
                auto suspended = std::move(*it);
                s.suspended_queue.erase(it);
                resource_slot slot = *slot_it;
                s.available_slots.erase(slot_it);

                s.pick_times[suspended.path] = std::chrono::steady_clock::now();

                // Cap the minimum batch size to MAX_ITERATIONS / 8
                int next_batch = std::max(MAX_ITERATIONS / 8, suspended.last_batch_size / 2);
                self->println("[INFO] Resuming solver for: {} (Device: {}, Stream: {}, Batch: {})", 
                              suspended.path, slot.device_id, slot.stream_id, next_batch);

                // Resume on the same stream ID. No update_stream_atom_v needed.
                self->mail(cg_next_step_atom_v, next_batch).send(suspended.solver);

                s.active_solvers.push_back(suspended.solver);
                s.task_resources[suspended.path] = slot;
                s.actor_batch_sizes[suspended.solver] = next_batch;
                resumed = true;
                break;
              }
            }
            if (!resumed) break;
          }
        }
    };

    spawn_next();
    
    return {
        [=](gpu_done_atom, const std::string& task_name, caf::actor solver, std::vector<float>& solution, solver_result_meta meta) {
            auto& s = self->state();

            if (meta.converged || meta.error_code != CG_SUCCESS) {
                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  end_time - s.start_times[task_name]).count();

                if (meta.converged && meta.iterations < MAX_ITERATIONS) {
                    s.tasks_succeeded++;
                    if (meta.iterations == 0)
                        self->println("[DONE] {}: Initial guess satisfied tolerance ({} ms).", task_name, duration);
                    else
                        self->println("[DONE] {}: Converged in {} iterations ({} ms).", task_name, meta.iterations, duration);
                } else {
                    s.tasks_failed++;
                    std::string reason = (meta.error_code == CG_SUCCESS)
                                         ? "Maximum Iterations Reached"
                                         : to_string(static_cast<cg_error_type>(meta.error_code));
                    self->println("[FAIL] {}: {} (after {} iterations, {} ms).", 
                                  task_name, reason,
                                  meta.iterations,
                                  duration);
                }

                // Final active slice
                auto active_slice = std::chrono::duration<double, std::milli>(end_time - s.pick_times[task_name]).count();
                s.cumulative_active_ms[task_name] += active_slice;

                // We override pick_time in record_job to simulate a single continuous run that equals 
                // the actual time spent on the GPU.
                auto simulated_pick = end_time - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                std::chrono::duration<double, std::milli>(s.cumulative_active_ms[task_name]));

                record_job(task_name, s.enqueue_times[task_name], simulated_pick, end_time, meta.iterations, meta.converged);
                s.enqueue_times.erase(task_name);
                s.pick_times.erase(task_name);
                s.cumulative_active_ms.erase(task_name);

                auto it = std::find(s.active_solvers.begin(), s.active_solvers.end(), solver);
                if (it != s.active_solvers.end())
                    s.active_solvers.erase(it);
                s.start_times.erase(task_name);
                s.actor_batch_sizes.erase(solver);

                // Reclaim the device/stream slot and put it back in the pool
                auto res_it = s.task_resources.find(task_name);
                if (res_it != s.task_resources.end()) {
                    s.available_slots.push_back(res_it->second);
                    s.task_resources.erase(res_it);
                }

                spawn_next();

                if (s.active_solvers.empty() && s.queue.empty() && s.suspended_queue.empty()) {
                    self->println("All tasks in the pool have been processed.");

                    report_workload_stats();
                    self->quit();
                }
            } else if (meta.iterations == 0) {
                // Just finished initialization: trigger the first iteration batch immediately.
                self->mail(cg_next_step_atom_v, s.num_iterations).send(solver);
            } else {
                // Not done: Suspend the actor to allow others to use the stream
                auto it = std::find(s.active_solvers.begin(), s.active_solvers.end(), solver);
                
                // Record the time spent in this active slice before suspending
                auto active_slice = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.pick_times[task_name]).count();
                s.cumulative_active_ms[task_name] += active_slice;

                if (it != s.active_solvers.end())
                    s.active_solvers.erase(it);

                resource_slot slot = s.task_resources[task_name];
                s.available_slots.push_back(slot);
                s.task_resources.erase(task_name);

                int last_batch = s.actor_batch_sizes[solver];
                s.suspended_queue.push_back({solver, task_name, last_batch, slot.device_id, slot.stream_id});

                self->println("[INFO] Suspending solver for: {} (Reclaimed Device: {}, Stream: {})", 
                              task_name, slot.device_id, slot.stream_id);

                spawn_next();
            }
        }
       
    };
}
