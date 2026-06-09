#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/component-actors/all-component-actors.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <random>
#include "caf/actor_registry.hpp"
#include <deque>
#include <unordered_map>
#include <unordered_set>
//#include <caf/atoms.hpp>

using namespace caf;
using namespace std::chrono_literals;

// ─────────────────────────────────────────────────────────────────────────────
// Atoms
// ─────────────────────────────────────────────────────────────────────────────
CAF_BEGIN_TYPE_ID_BLOCK(mmul_benchmark, caf::id_block::cuda::end)
    CAF_ADD_ATOM(mmul_benchmark, get_work_atom)
    CAF_ADD_ATOM(mmul_benchmark, task_done_atom)
    CAF_ADD_ATOM(mmul_benchmark, release_memory_atom)
    CAF_ADD_ATOM(mmul_benchmark, request_work_atom)
    CAF_ADD_ATOM(mmul_benchmark, worker_done_atom)
    CAF_ADD_ATOM(mmul_benchmark, refill_buffer_atom)
CAF_END_TYPE_ID_BLOCK(mmul_benchmark)

// Command runners for GPU operations
caf::cuda::command_runner<> mmul_command;
using mmul_kernel_t = caf::cuda::command_runner<caf::cuda::mem_ptr<int>, caf::cuda::mem_ptr<int>, out<int>, in<int>>;
mmul_kernel_t mmul_kernel;

struct MatrixPool {
    std::unordered_map<int, std::vector<int>> A;
    std::unordered_map<int, std::vector<int>> B;
};



MatrixPool create_matrix_pool_random(
    int num_sizes,
    int min_N,
    int max_N,
    unsigned int seed
) {
    MatrixPool pool;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(min_N, max_N);

    std::unordered_set<int> used;

    while (used.size() < static_cast<size_t>(num_sizes)) {
        int N = dist(rng);
        if (used.insert(N).second) {
            pool.A[N] = std::vector<int>(N * N, 1);
            pool.B[N] = std::vector<int>(N * N, 1);
        }
    }

    return pool;
}

// ---------------------------- GLOBAL TASK POOL ----------------------------
// The central source of truth for work. Implements a pull-based model.
struct task_pool_state {
    std::vector<int> tasks;
    size_t next_task_idx = 0;
};

caf::behavior global_task_pool(caf::stateful_actor<task_pool_state>* self, std::vector<int> tasks) {
    self->state().tasks = std::move(tasks);
    return {
        [=](get_work_atom, size_t batch_size) -> result<std::vector<int>> {
            auto& st = self->state();
            if (st.next_task_idx >= st.tasks.size())
                return sec::end_of_stream;
            size_t count = std::min(batch_size, st.tasks.size() - st.next_task_idx);
            std::vector<int> batch(st.tasks.begin() + st.next_task_idx, 
                                   st.tasks.begin() + st.next_task_idx + count);
            st.next_task_idx += count;
            return batch;
        }
    };
}

// ---------------------------- DEVICE/GPU ACTOR ----------------------------
// Manages memory for a specific GPU and steals (pulls) work from the Global Pool.
struct device_actor_state {
    MatrixPool pool;
    caf::actor global_pool;
    std::deque<int> local_tasks; // Local buffer to keep GPU busy
    size_t total_device_memory_bytes = 0;
    size_t current_allocated_memory_bytes = 0;
    int active_workers = 0;
    int device_id = -1;
    size_t batch_size = 0;
    size_t low_water_mark = 0;
    bool fetching = false;
};

caf::behavior gpu_device_actor(caf::stateful_actor<device_actor_state>* self,
                               MatrixPool pool, caf::actor global_pool, int num_workers, int dev_id, int max_in_flight) {
    self->state().pool = std::move(pool);
    self->state().global_pool = global_pool;
    self->state().device_id = dev_id;
    self->state().active_workers = num_workers;

    // Dynamically calculate prefetch markers based on the total pipeline capacity
    self->state().low_water_mark = static_cast<size_t>(num_workers * max_in_flight);
    self->state().batch_size = self->state().low_water_mark * 2;

    caf::cuda::manager& mgr = caf::cuda::manager::get();
    caf::cuda::device_ptr dev_obj = mgr.find_device(dev_id);
    if (dev_obj) {
        self->state().total_device_memory_bytes = dev_obj->total_memory_bytes();
    }

    // Helper to refill the local task buffer from the global pool
    auto refill = [=]() {
        auto& st = self->state();
        if (st.fetching || st.local_tasks.size() >= st.low_water_mark + st.batch_size)
            return;

        st.fetching = true;
        self->mail(get_work_atom_v, (size_t)st.batch_size).request(st.global_pool, infinite).then(
            [=](std::vector<int>& batch) {
                auto& st_inner = self->state();
                for (int N : batch)
                    st_inner.local_tasks.push_back(N);
                st_inner.fetching = false;
                if (st_inner.local_tasks.size() < st_inner.low_water_mark)
                    self->mail(refill_buffer_atom_v).send(self);
            },
            [=](error& err) {
                self->state().fetching = false;
            }
        );
    };

    return {
        [=](refill_buffer_atom) {
            refill();
        },
        [=](get_work_atom) -> caf::result<int, in<int>, in<int>> {
            auto& st = self->state();

            // If we have tasks locally, satisfy the request immediately
            if (!st.local_tasks.empty()) {
                int N = st.local_tasks.front();
                size_t memory_needed = (size_t)N * N * sizeof(int) * 3;
                if (st.current_allocated_memory_bytes + memory_needed > st.total_device_memory_bytes)
                    return make_error(sec::runtime_error, "Device Actor: Not enough memory");
                
                st.local_tasks.pop_front();
                st.current_allocated_memory_bytes += memory_needed;
                
                // Proactively steal more work if the buffer is getting low
                if (st.local_tasks.size() < st.low_water_mark)
                    refill();

                return {N, caf::cuda::create_in_arg(st.pool.A[N]), 
                           caf::cuda::create_in_arg(st.pool.B[N])};
            }

            // Buffer empty: must fetch from global pool reactively
            auto promise = self->make_response_promise<int, in<int>, in<int>>();
            self->mail(get_work_atom_v, (size_t)st.batch_size).request(st.global_pool, infinite).then(
                [=](std::vector<int>& batch) mutable {
                    auto& st_inner = self->state();
                    int N = batch.front();
                    for(size_t i = 1; i < batch.size(); ++i) st_inner.local_tasks.push_back(batch[i]);
                    
                    size_t needed = (size_t)N * N * sizeof(int) * 3;
                    st_inner.current_allocated_memory_bytes += needed;
                    promise.deliver(N, caf::cuda::create_in_arg(st_inner.pool.A[N]), 
                                       caf::cuda::create_in_arg(st_inner.pool.B[N]));
                },
                [=](error& err) mutable { promise.deliver(err); }
            );
            return promise;
        },
        [=](release_memory_atom, int N_completed) {
            auto& st = self->state();
            size_t memory_released = (size_t)N_completed * N_completed * sizeof(int) * 3;
            st.current_allocated_memory_bytes -= memory_released;
            refill(); // Try to get more work now that memory is free
        },
        [=](worker_done_atom) {
            auto& st = self->state();
            if (--st.active_workers <= 0) {
                self->quit();
            }
        }
    };
}

// ---------------------------- WORKER ACTOR ----------------------------
// Manages 1 stream and pulls work from the Device Actor.
struct worker_state {
    int device_id;
    std::vector<int> stream_ids;
    caf::cuda::program_ptr program;
    caf::actor device_actor;
    caf::actor supervisor;
    int max_in_flight_tasks;
    int in_flight_tasks_count = 0;
    bool draining = false;
};

caf::behavior mmul_worker_fun(caf::stateful_actor<worker_state>* self,
                              caf::actor supervisor, caf::actor device_actor, caf::cuda::program_ptr program,
                              int dev_id, std::vector<int> stream_ids, int max_in_flight_tasks) {
    self->state().supervisor = supervisor;
    self->state().device_actor = device_actor;
    self->state().program = program;
    self->state().device_id = dev_id;
    self->state().stream_ids = std::move(stream_ids);
    self->state().max_in_flight_tasks = max_in_flight_tasks;

    // Trigger initial work requests up to max_in_flight_tasks
    for (int i = 0; i < max_in_flight_tasks; ++i) {
        self->mail(request_work_atom_v).send(self);
    }

    return {
        [=](request_work_atom) {
            auto& st = self->state();
            if (st.in_flight_tasks_count >= st.max_in_flight_tasks || st.draining) {
                return; // Already at max capacity, don't request more yet
            }

            st.in_flight_tasks_count++; // Mark as pending immediately
            self->mail(get_work_atom_v).request(st.device_actor, infinite).then(
                [=](int N, in<int> matrixA, in<int> matrixB) mutable {
                    auto& st = self->state(); // Access state via self
                    int s_h2d = st.stream_ids[0];
                    int s_ker = st.stream_ids[1];
                    int s_d2h = st.stream_ids[2];

                    // Create fresh events for this specific task pipeline
                    auto h2d_done = mmul_command.create_event(st.device_id);
                    auto kernel_done = mmul_command.create_event(st.device_id);

                    // Stage 1: H2D Transfer
                    auto arg1 = mmul_command.transfer_memory(st.device_id, s_h2d, std::move(matrixA));
                    auto arg2 = mmul_command.transfer_memory(st.device_id, s_h2d, std::move(matrixB));
                    mmul_command.record_event(h2d_done, s_h2d, st.device_id);
                  
                    const int THREADS = 32;
                    const int BLOCKS = (N + THREADS - 1) / THREADS;
                    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

                    // Stage 2: Kernel Execution (Wait for H2D to finish)
                    mmul_command.wait_event(h2d_done, s_ker, st.device_id);
                    auto result = mmul_kernel.run_async(st.program, dims, s_ker, 0, st.device_id,
                                                       arg1, arg2,
                                                       caf::cuda::create_out_arg<int>(N * N),
                                                       caf::cuda::create_in_arg<int>(N));
                    mmul_command.record_event(kernel_done, s_ker, st.device_id);
                      mmul_command.add_callback(s_ker, st.device_id, [arg1, arg2, self]() {
                        anon_mail(arg1,arg2).send(self);
                    });


                    // Stage 3: D2H Copyback (Wait for Kernel to finish)
                    mmul_command.wait_event(kernel_done, s_d2h, st.device_id);
                    auto bufferC = std::get<2>(result);
                    auto self_hdl = caf::actor_cast<caf::actor>(self);

                  
                    mmul_command.copy_to_host_async(bufferC, s_d2h,
                      [self_hdl, N_task = N](std::vector<int>&&) {
                        caf::anon_mail(task_done_atom_v, N_task).send(self_hdl); // Pass N back to self
                    });
                },
                [=](error& err) {
                    auto& st = self->state();
                    st.in_flight_tasks_count--; // Revert pending status on failure
                    if (err == sec::runtime_error) {
                        // Not enough memory, retry after a delay
                        self->println("Worker (dev:{}): Not enough memory, retrying for work...", st.device_id);
                        self->delayed_anon_send(self, 100ms, request_work_atom_v);
                    } else if (err == sec::end_of_stream) {
                        st.draining = true; // Mark as draining, let in-flight finish
                        if (st.in_flight_tasks_count == 0) {
                            self->mail(worker_done_atom_v).send(st.device_actor);
                            for (auto id : st.stream_ids)
                                mmul_command.release_stream_for_actor(id);
                            self->quit();
                        }
                    }
                }
            );
        },
        [=](caf::cuda::mem_ptr<int> matrixA, caf::cuda::mem_ptr<int> matrixB) {
            mmul_command.free_memory(matrixA, stream_ids[1]);
            mmul_command.free_memory(matrixB, stream_ids[1]);
        },
        [=](task_done_atom, int N_completed) {
            auto& st = self->state();
            st.in_flight_tasks_count--; // Decrement count
            self->mail(1).send(st.supervisor); // Notify supervisor
            self->mail(release_memory_atom_v, N_completed).send(st.device_actor); // Release memory
            
            if (st.draining && st.in_flight_tasks_count == 0) {
                self->mail(worker_done_atom_v).send(st.device_actor);
                for (auto id : st.stream_ids)
                    mmul_command.release_stream_for_actor(id);
                self->quit();
            } else if (!st.draining) {
                self->mail(request_work_atom_v).send(self); // Request next task if capacity allows
            }
        }
    };
}

// ---------------------------- SUPERVISOR ACTOR ----------------------------
struct supervisor_actor_state {
    int total_tasks;
    int completed = 0;
    std::chrono::steady_clock::time_point start_time;
};

caf::behavior supervisor_actor_fun(
    caf::stateful_actor<supervisor_actor_state>* self,
    int total_tasks,
    int workers_per_gpu,
    int max_in_flight_tasks_per_worker,
    MatrixPool pool,
    std::vector<int> Ns
    ) {
    self->state().total_tasks = total_tasks;
    self->state().start_time = std::chrono::steady_clock::now();

    auto pool_actor = self->spawn(global_task_pool, std::move(Ns));

    caf::cuda::manager& mgr = caf::cuda::manager::get();
    int num_gpus = mgr.get_num_devices();
    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
    
    int next_stream_base = 0;

    for (int i = 0; i < num_gpus; ++i) {
        auto broker = self->spawn(gpu_device_actor, pool, pool_actor, workers_per_gpu, i, max_in_flight_tasks_per_worker);
        
        for (int j = 0; j < workers_per_gpu; ++j) {
            std::vector<int> streams = {next_stream_base, next_stream_base + 1, next_stream_base + 2};
            self->spawn(mmul_worker_fun, self, broker, program, i, streams, max_in_flight_tasks_per_worker);
            next_stream_base += 3;
        }
    }

    return {
        [=](int done) {
            self->state().completed += done;
            if (self->state().completed >= self->state().total_tasks) {
                auto end_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> total_time = end_time - self->state().start_time;

                std::cout << "\n===== BENCHMARK COMPLETE =====\n";
                std::cout << "Tasks: " << self->state().total_tasks << "\n";
                std::cout << "Runtime: " << total_time.count() << " s\n";
                
                caf::cuda::manager::shutdown();
                self->quit();
            }
        }
    };
}

template <class Fn>
double time_run(Fn&& fn) {
  auto start = std::chrono::steady_clock::now();
  fn();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

void run_mmul_random_scaling_tests(caf::actor_system& sys,
                                  caf::cuda::manager_config man_config) {

    const int min_N = 32;
    const int max_N = 2048;
    const int num_sizes = 10;

    const int workers_per_gpu = 4; // Admission control: only 16 concurrent tasks per GPU
    const int max_in_flight_tasks_per_worker = 2; // Each worker keeps 2 tasks in flight

    const std::vector<int> actor_counts = {
	  1,30000,40000,50000
    };

    // Generate deterministic random pool once
    MatrixPool pool = create_matrix_pool_random(
        num_sizes,
        min_N,
        max_N,
        42  // fixed seed
    );

    //scheduler
    caf::cuda::manager_config scheduler_off(false);
    for (int num_tasks_for_this_run : actor_counts) {
	    // Initialize CUDA manager
	    caf::cuda::manager::init(sys, scheduler_off);
        std::cout << "=====================================\n";
        std::cout << "Random Scaling | actors=" << num_tasks_for_this_run << "\n";

        // Precompute all task Ns for this run
        std::vector<int> sizes;
        for (const auto& [N, _] : pool.A) sizes.push_back(N);

        std::vector<int> Ns_for_this_run;
        Ns_for_this_run.reserve(num_tasks_for_this_run);

        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, sizes.size() - 1);
        for (int i = 0; i < num_tasks_for_this_run; ++i)
	        Ns_for_this_run.push_back(sizes[dist(rng)]);

        // Execute the supervisor which manages the asynchronous workload
        double elapsed = time_run([&]() {

            auto sup = sys.spawn(
                supervisor_actor_fun,
                (int)Ns_for_this_run.size(), // total_tasks
                workers_per_gpu,
                max_in_flight_tasks_per_worker,
                pool,
		Ns_for_this_run
            );

	      sys.await_all_actors_done();
	    });

        caf::cuda::manager::shutdown();
    }
}
void caf_main(caf::actor_system& sys) {
  caf::cuda::manager_config man_config(false);
  run_mmul_random_scaling_tests(sys, man_config);
}
CAF_MAIN(id_block::mmul_benchmark)
