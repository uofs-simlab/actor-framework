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
  int max_active = 1;
  int num_iterations = MAX_ITERATIONS / 2;
  int num_gpus = 0;
  int tasks_succeeded = 0;
  int tasks_failed = 0;
  std::chrono::steady_clock::time_point benchmark_start;
};

inline caf::behavior supervisor_actor(caf::stateful_actor<supervisor_state>* self, 
                                      std::vector<MatrixTask> tasks, 
                                      int initial_max_active, 
                                      std::chrono::steady_clock::time_point start_time) {
    auto& st = self->state();
    st.queue.insert(st.queue.end(), std::make_move_iterator(tasks.begin()), std::make_move_iterator(tasks.end()));
    st.max_active = initial_max_active;
    st.benchmark_start = start_time;
    st.num_gpus = caf::cuda::manager::get().get_num_devices();

    for (int s = 0; s < 32; ++s) {
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

            auto solver = self->spawn(fault_tolerant_cg_actor<float>,
                                    path, task.data,
                                    caf::cuda::create_in_arg(task.data->row_ptr),
                                    caf::cuda::create_in_arg(task.data->col_indices),
                                    caf::cuda::create_in_arg(task.data->values),
                                    caf::cuda::create_in_arg(task.data->b),
                                    caf::cuda::create_in_out_arg(task.data->x_guess),
                                    (int)task.data->row_ptr.size() - 1,
                                    (int)task.data->values.size(),
                                    1e-5f, MAX_ITERATIONS, slot.device_id, slot.stream_id, caf::actor_cast<caf::actor>(self));

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
                int next_batch = std::max(MAX_ITERATIONS / 8, suspended.last_batch_size / 2);
                self->mail(cg_next_step_atom, next_batch).send(suspended.solver);
                s.active_solvers.push_back(suspended.solver);
                s.task_resources[suspended.path] = slot;
                s.actor_batch_sizes[suspended.solver] = next_batch;
                resumed = true; break;
              }
            }
            if (!resumed) break;
          }
        }
    };

    spawn_next();
    
    return {
        [=](gpu_done_atom, const std::string& task_name, caf::actor solver, std::vector<float>& solution, caf::cuda::solver_result_meta meta) {
            auto& s = self->state();
            if (meta.converged || meta.error_code != CG_SUCCESS) {
                auto end_time = std::chrono::steady_clock::now();
                if (meta.converged && meta.iterations < MAX_ITERATIONS) s.tasks_succeeded++;
                else s.tasks_failed++;

                auto active_slice = std::chrono::duration<double, std::milli>(end_time - s.pick_times[task_name]).count();
                s.cumulative_active_ms[task_name] += active_slice;
                auto simulated_pick = end_time - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                std::chrono::duration<double, std::milli>(s.cumulative_active_ms[task_name]));

                record_job(task_name, s.enqueue_times[task_name], simulated_pick, end_time, meta.iterations, meta.converged);
                auto it = std::find(s.active_solvers.begin(), s.active_solvers.end(), solver);
                if (it != s.active_solvers.end()) s.active_solvers.erase(it);
                auto res_it = s.task_resources.find(task_name);
                if (res_it != s.task_resources.end()) {
                    s.available_slots.push_back(res_it->second);
                    s.task_resources.erase(res_it);
                }
                spawn_next();
                if (s.active_solvers.empty() && s.queue.empty() && s.suspended_queue.empty()) {
                    report_workload_stats();
                    self->quit();
                }
            } else if (meta.iterations == 0) {
                self->mail(cg_next_step_atom, s.num_iterations).send(solver);
            } else {
                auto it = std::find(s.active_solvers.begin(), s.active_solvers.end(), solver);
                auto active_slice = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.pick_times[task_name]).count();
                s.cumulative_active_ms[task_name] += active_slice;
                if (it != s.active_solvers.end()) s.active_solvers.erase(it);
                resource_slot slot = s.task_resources[task_name];
                s.available_slots.push_back(slot); s.task_resources.erase(task_name);
                s.suspended_queue.push_back({solver, task_name, s.actor_batch_sizes[solver], slot.device_id, slot.stream_id});
                spawn_next();
            }
        }
    };
}